//! Split an OSTree commit into separate chunks

// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::borrow::{Borrow, Cow};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write;
use std::hash::{Hash, Hasher};
use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::Instant;

use crate::objectsource::{ContentID, ObjectMeta, ObjectMetaMap, ObjectSourceMeta};
use crate::objgv::*;
use crate::statistics;
use anyhow::{anyhow, Result};
use camino::Utf8PathBuf;
use gvariant::aligned_bytes::TryAsAligned;
use gvariant::{Marker, Structure};
use ostree::{gio, glib};
use serde::{Deserialize, Serialize};

/// Maximum number of layers (chunks) we will use.
// We take half the limit of 128.
// https://github.com/ostreedev/ostree-rs-ext/issues/69
pub(crate) const MAX_CHUNKS: u32 = 64;

type RcStr = Rc<str>;
pub(crate) type ChunkMapping = BTreeMap<RcStr, (u64, Vec<Utf8PathBuf>)>;

#[derive(Debug, Default)]
pub(crate) struct Chunk {
    pub(crate) name: String,
    pub(crate) content: ChunkMapping,
    pub(crate) size: u64,
    pub(crate) packages: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
/// Object metadata, but with additional size data
pub struct ObjectSourceMetaSized {
    /// The original metadata
    #[serde(flatten)]
    meta: ObjectSourceMeta,
    /// Total size of associated objects
    size: u64,
}

impl Hash for ObjectSourceMetaSized {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.meta.identifier.hash(state);
    }
}

impl Eq for ObjectSourceMetaSized {}

impl PartialEq for ObjectSourceMetaSized {
    fn eq(&self, other: &Self) -> bool {
        self.meta.identifier == other.meta.identifier
    }
}

/// Extend content source metadata with sizes.
#[derive(Debug)]
pub struct ObjectMetaSized {
    /// Mapping from content object to source.
    pub map: ObjectMetaMap,
    /// Computed sizes of each content source
    pub sizes: Vec<ObjectSourceMetaSized>,
}

impl ObjectMetaSized {
    /// Given object metadata and a repo, compute the size of each content source.
    pub fn compute_sizes(repo: &ostree::Repo, meta: ObjectMeta) -> Result<ObjectMetaSized> {
        let cancellable = gio::Cancellable::NONE;
        // Destructure into component parts; we'll create the version with sizes
        let map = meta.map;
        let mut set = meta.set;
        // Maps content id -> total size of associated objects
        let mut sizes = HashMap::<&str, u64>::new();
        // Populate two mappings above, iterating over the object -> contentid mapping
        for (checksum, contentid) in map.iter() {
            let finfo = repo.query_file(checksum, cancellable)?.0;
            let sz = sizes.entry(contentid).or_default();
            *sz += finfo.size() as u64;
        }
        // Combine data from sizes and the content mapping.
        let sized: Result<Vec<_>> = sizes
            .into_iter()
            .map(|(id, size)| -> Result<ObjectSourceMetaSized> {
                set.take(id)
                    .ok_or_else(|| anyhow!("Failed to find {} in content set", id))
                    .map(|meta| ObjectSourceMetaSized { meta, size })
            })
            .collect();
        let mut sizes = sized?;
        sizes.sort_by(|a, b| b.size.cmp(&a.size));
        Ok(ObjectMetaSized { map, sizes })
    }
}

/// How to split up an ostree commit into "chunks" - designed to map to container image layers.
#[derive(Debug, Default)]
pub struct Chunking {
    pub(crate) metadata_size: u64,
    pub(crate) remainder: Chunk,
    pub(crate) chunks: Vec<Chunk>,

    pub(crate) max: u32,

    processed_mapping: bool,
    /// Number of components (e.g. packages) provided originally
    pub(crate) n_provided_components: u32,
    /// The above, but only ones with non-zero size
    pub(crate) n_sized_components: u32,
}

#[derive(Default)]
struct Generation {
    path: Utf8PathBuf,
    metadata_size: u64,
    dirtree_found: BTreeSet<RcStr>,
    dirmeta_found: BTreeSet<RcStr>,
}

fn push_dirmeta(repo: &ostree::Repo, gen: &mut Generation, checksum: &str) -> Result<()> {
    if gen.dirtree_found.contains(checksum) {
        return Ok(());
    }
    let checksum = RcStr::from(checksum);
    gen.dirmeta_found.insert(RcStr::clone(&checksum));
    let child_v = repo.load_variant(ostree::ObjectType::DirMeta, checksum.borrow())?;
    gen.metadata_size += child_v.data_as_bytes().as_ref().len() as u64;
    Ok(())
}

fn push_dirtree(
    repo: &ostree::Repo,
    gen: &mut Generation,
    checksum: &str,
) -> Result<glib::Variant> {
    let child_v = repo.load_variant(ostree::ObjectType::DirTree, checksum)?;
    if !gen.dirtree_found.contains(checksum) {
        gen.metadata_size += child_v.data_as_bytes().as_ref().len() as u64;
    } else {
        let checksum = RcStr::from(checksum);
        gen.dirtree_found.insert(checksum);
    }
    Ok(child_v)
}

fn generate_chunking_recurse(
    repo: &ostree::Repo,
    gen: &mut Generation,
    chunk: &mut Chunk,
    dt: &glib::Variant,
) -> Result<()> {
    let dt = dt.data_as_bytes();
    let dt = dt.try_as_aligned()?;
    let dt = gv_dirtree!().cast(dt);
    let (files, dirs) = dt.to_tuple();
    // A reusable buffer to avoid heap allocating these
    let mut hexbuf = [0u8; 64];
    for file in files {
        let (name, csum) = file.to_tuple();
        let fpath = gen.path.join(name.to_str());
        hex::encode_to_slice(csum, &mut hexbuf)?;
        let checksum = std::str::from_utf8(&hexbuf)?;
        let meta = repo.query_file(checksum, gio::Cancellable::NONE)?.0;
        let size = meta.size() as u64;
        let entry = chunk.content.entry(RcStr::from(checksum)).or_default();
        entry.0 = size;
        let first = entry.1.is_empty();
        if first {
            chunk.size += size;
        }
        entry.1.push(fpath);
    }
    for item in dirs {
        let (name, contents_csum, meta_csum) = item.to_tuple();
        let name = name.to_str();
        // Extend our current path
        gen.path.push(name);
        hex::encode_to_slice(contents_csum, &mut hexbuf)?;
        let checksum_s = std::str::from_utf8(&hexbuf)?;
        let dirtree_v = push_dirtree(repo, gen, checksum_s)?;
        generate_chunking_recurse(repo, gen, chunk, &dirtree_v)?;
        drop(dirtree_v);
        hex::encode_to_slice(meta_csum, &mut hexbuf)?;
        let checksum_s = std::str::from_utf8(&hexbuf)?;
        push_dirmeta(repo, gen, checksum_s)?;
        // We did a push above, so pop must succeed.
        assert!(gen.path.pop());
    }
    Ok(())
}

impl Chunk {
    fn new(name: &str) -> Self {
        Chunk {
            name: name.to_string(),
            ..Default::default()
        }
    }

    fn move_obj(&mut self, dest: &mut Self, checksum: &str) -> bool {
        // In most cases, we expect the object to exist in the source.  However, it's
        // conveneient here to simply ignore objects which were already moved into
        // a chunk.
        if let Some((name, (size, paths))) = self.content.remove_entry(checksum) {
            let v = dest.content.insert(name, (size, paths));
            debug_assert!(v.is_none());
            self.size -= size;
            dest.size += size;
            true
        } else {
            false
        }
    }
}

impl Chunking {
    /// Generate an initial single chunk.
    pub fn new(repo: &ostree::Repo, rev: &str) -> Result<Self> {
        // Find the target commit
        let rev = repo.require_rev(rev)?;

        // Load and parse the commit object
        let (commit_v, _) = repo.load_commit(&rev)?;
        let commit_v = commit_v.data_as_bytes();
        let commit_v = commit_v.try_as_aligned()?;
        let commit = gv_commit!().cast(commit_v);
        let commit = commit.to_tuple();

        // Load it all into a single chunk
        let mut gen = Generation {
            path: Utf8PathBuf::from("/"),
            ..Default::default()
        };
        let mut chunk: Chunk = Default::default();

        // Find the root directory tree
        let contents_checksum = &hex::encode(commit.6);
        let contents_v = repo.load_variant(ostree::ObjectType::DirTree, contents_checksum)?;
        push_dirtree(repo, &mut gen, contents_checksum)?;
        let meta_checksum = &hex::encode(commit.7);
        push_dirmeta(repo, &mut gen, meta_checksum.as_str())?;

        generate_chunking_recurse(repo, &mut gen, &mut chunk, &contents_v)?;

        let chunking = Chunking {
            metadata_size: gen.metadata_size,
            remainder: chunk,
            ..Default::default()
        };
        Ok(chunking)
    }

    /// Generate a chunking from an object mapping.
    pub fn from_mapping(
        repo: &ostree::Repo,
        rev: &str,
        meta: ObjectMetaSized,
        max_layers: &Option<NonZeroU32>,
        prior_build_metadata: &Option<Vec<Vec<String>>>,
    ) -> Result<Self> {
        let mut r = Self::new(repo, rev)?;
        r.process_mapping(meta, max_layers, prior_build_metadata)?;
        Ok(r)
    }

    fn remaining(&self) -> u32 {
        self.max.saturating_sub(self.chunks.len() as u32)
    }

    /// Given metadata about which objects are owned by a particular content source,
    /// generate chunks that group together those objects.
    #[allow(clippy::or_fun_call)]
    pub fn process_mapping(
        &mut self,
        meta: ObjectMetaSized,
        max_layers: &Option<NonZeroU32>,
        prior_build_metadata: &Option<Vec<Vec<String>>>,
    ) -> Result<()> {
        self.max = max_layers
            .unwrap_or(NonZeroU32::new(MAX_CHUNKS).unwrap())
            .get();

        let sizes = &meta.sizes;
        // It doesn't make sense to handle multiple mappings
        assert!(!self.processed_mapping);
        self.processed_mapping = true;
        let remaining = self.remaining();
        if remaining == 0 {
            return Ok(());
        }

        // Reverses `contentmeta.map` i.e. contentid -> Vec<checksum>
        let mut rmap = HashMap::<ContentID, Vec<&String>>::new();
        for (checksum, contentid) in meta.map.iter() {
            rmap.entry(Rc::clone(contentid)).or_default().push(checksum);
        }

        // Safety: Let's assume no one has over 4 billion components.
        self.n_provided_components = meta.sizes.len().try_into().unwrap();
        self.n_sized_components = sizes
            .iter()
            .filter(|v| v.size > 0)
            .count()
            .try_into()
            .unwrap();

        // TODO: Compute bin packing in a better way
        let start = Instant::now();
        let packing = basic_packing(
            sizes,
            NonZeroU32::new(self.max).unwrap(),
            prior_build_metadata,
        );
        let duration = start.elapsed();
        println!("Time elapsed in packing: {:#?}", duration);

        for bin in packing.into_iter() {
            let name = match bin.len() {
                0 => Cow::Borrowed("Reserved for new packages"),
                1 => {
                    let first = bin[0];
                    let first_name = &*first.meta.identifier;
                    Cow::Borrowed(first_name)
                }
                2..=5 => {
                    let first = bin[0];
                    let first_name = &*first.meta.identifier;
                    let r = bin.iter().map(|v| &*v.meta.identifier).skip(1).fold(
                        String::from(first_name),
                        |mut acc, v| {
                            write!(acc, " and {}", v).unwrap();
                            acc
                        },
                    );
                    Cow::Owned(r)
                }
                n => Cow::Owned(format!("{n} components")),
            };
            let mut chunk = Chunk::new(&*name);
            chunk.packages = bin.iter().map(|v| String::from(&*v.meta.name)).collect();
            for szmeta in bin {
                for &obj in rmap.get(&szmeta.meta.identifier).unwrap() {
                    self.remainder.move_obj(&mut chunk, obj.as_str());
                }
            }
            self.chunks.push(chunk);
        }

        assert_eq!(self.remainder.content.len(), 0);

        Ok(())
    }

    pub(crate) fn take_chunks(&mut self) -> Vec<Chunk> {
        let mut r = Vec::new();
        std::mem::swap(&mut self.chunks, &mut r);
        r
    }

    /// Print information about chunking to standard output.
    pub fn print(&self) {
        println!("Metadata: {}", glib::format_size(self.metadata_size));
        if self.n_provided_components > 0 {
            println!(
                "Components: provided={} sized={}",
                self.n_provided_components, self.n_sized_components
            );
        }
        for (n, chunk) in self.chunks.iter().enumerate() {
            let sz = glib::format_size(chunk.size);
            println!(
                "Chunk {}: \"{}\": objects:{} size:{}",
                n,
                chunk.name,
                chunk.content.len(),
                sz
            );
        }
        if !self.remainder.content.is_empty() {
            let sz = glib::format_size(self.remainder.size);
            println!(
                "Remainder: \"{}\": objects:{} size:{}",
                self.remainder.name,
                self.remainder.content.len(),
                sz
            );
        }
    }
}

#[cfg(test)]
fn components_size(components: &[&ObjectSourceMetaSized]) -> u64 {
    components.iter().map(|k| k.size).sum()
}

/// Compute the total size of a packing
#[cfg(test)]
fn packing_size(packing: &[Vec<&ObjectSourceMetaSized>]) -> u64 {
    packing.iter().map(|v| components_size(v)).sum()
}

///Given a certain threshold, divide a list of packages into all combinations
///of (high, medium, low) size and (high,medium,low) using the following
///outlier detection methods:
///- Median and Median Absolute Deviation Method
///     Aggressively detects outliers in size and classifies them by
///     high, medium, low. The high size and low size are separate partitions
///     and deserve bins of their own
///- Mean and Standard Deviation Method
///     The medium partition from the previous step is less aggressively
///     classified by using mean for both size and frequency

//Assumes components is sorted by descending size
fn get_partitions_with_threshold(
    components: Vec<&ObjectSourceMetaSized>,
    limit_hs_bins: usize,
    threshold: f64,
) -> Option<BTreeMap<String, Vec<&ObjectSourceMetaSized>>> {
    let mut partitions: BTreeMap<String, Vec<&ObjectSourceMetaSized>> = BTreeMap::new();
    let mut med_size: Vec<&ObjectSourceMetaSized> = Vec::new();
    let mut high_size: Vec<&ObjectSourceMetaSized> = Vec::new();

    let mut sizes: Vec<u64> = components.iter().map(|a| a.size).collect();
    let (median_size, mad_size) = statistics::median_absolute_deviation(&mut sizes)?;

    //Avoids lower limit being negative
    let size_low_limit = 0.5 * f64::abs(median_size - threshold * mad_size);
    let size_high_limit = median_size + threshold * mad_size;

    for pkg in components {
        let size = pkg.size as f64;

        //high size (hs)
        if size >= size_high_limit {
            high_size.push(pkg);
        }
        //low size (ls)
        else if size <= size_low_limit {
            partitions
                .entry("2ls".to_string())
                .and_modify(|bin| bin.push(pkg))
                .or_insert_with(|| vec![pkg]);
        }
        //medium size (ms)
        else {
            med_size.push(pkg);
        }
    }

    //Extra hs packages
    let mut remaining_pkgs: Vec<_> = high_size.drain(limit_hs_bins..).collect();
    assert_eq!(high_size.len(), limit_hs_bins);

    //Concatenate extra hs packages + med_sizes to keep it descending sorted
    remaining_pkgs.append(&mut med_size);
    partitions.insert("1hs".to_string(), high_size);

    //Ascending sorted by frequency, so each partition within ms is freq sorted
    remaining_pkgs.sort_by(|a, b| {
        a.meta
            .change_frequency
            .partial_cmp(&b.meta.change_frequency)
            .unwrap()
    });
    let med_sizes: Vec<u64> = remaining_pkgs.iter().map(|a| a.size).collect();
    let med_frequencies: Vec<u64> = remaining_pkgs
        .iter()
        .map(|a| a.meta.change_frequency.into())
        .collect();

    let med_mean_freq = statistics::mean(&med_frequencies)?;
    let med_stddev_freq = statistics::std_deviation(&med_frequencies)?;
    let med_mean_size = statistics::mean(&med_sizes)?;
    let med_stddev_size = statistics::std_deviation(&med_sizes)?;

    //Avoids lower limit being negative
    let med_freq_low_limit = 0.5f64 * f64::abs(med_mean_freq - threshold * med_stddev_freq);
    let med_freq_high_limit = med_mean_freq + threshold * med_stddev_freq;
    let med_size_low_limit = 0.5f64 * f64::abs(med_mean_size - threshold * med_stddev_size);
    let med_size_high_limit = med_mean_size + threshold * med_stddev_size;

    for pkg in remaining_pkgs {
        let size = pkg.size as f64;
        let freq = pkg.meta.change_frequency as f64;

        //low frequency, high size
        if (freq <= med_freq_low_limit) && (size >= med_size_high_limit) {
            partitions
                .entry("lf_hs".to_string())
                .and_modify(|bin| bin.push(pkg))
                .or_insert_with(|| vec![pkg]);
        }
        //medium frequency, high size
        else if (freq < med_freq_high_limit)
            && (freq > med_freq_low_limit)
            && (size >= med_size_high_limit)
        {
            partitions
                .entry("mf_hs".to_string())
                .and_modify(|bin| bin.push(pkg))
                .or_insert_with(|| vec![pkg]);
        }
        //high frequency, high size
        else if (freq >= med_freq_high_limit) && (size >= med_size_high_limit) {
            partitions
                .entry("hf_hs".to_string())
                .and_modify(|bin| bin.push(pkg))
                .or_insert_with(|| vec![pkg]);
        }
        //low frequency, medium size
        else if (freq <= med_freq_low_limit)
            && (size < med_size_high_limit)
            && (size > med_size_low_limit)
        {
            partitions
                .entry("lf_ms".to_string())
                .and_modify(|bin| bin.push(pkg))
                .or_insert_with(|| vec![pkg]);
        }
        //medium frequency, medium size
        else if (freq < med_freq_high_limit)
            && (freq > med_freq_low_limit)
            && (size < med_size_high_limit)
            && (size > med_size_low_limit)
        {
            partitions
                .entry("mf_ms".to_string())
                .and_modify(|bin| bin.push(pkg))
                .or_insert_with(|| vec![pkg]);
        }
        //high frequency, medium size
        else if (freq >= med_freq_high_limit)
            && (size < med_size_high_limit)
            && (size > med_size_low_limit)
        {
            partitions
                .entry("hf_ms".to_string())
                .and_modify(|bin| bin.push(pkg))
                .or_insert_with(|| vec![pkg]);
        }
        //low frequency, low size
        else if (freq <= med_freq_low_limit) && (size <= med_size_low_limit) {
            partitions
                .entry("lf_ls".to_string())
                .and_modify(|bin| bin.push(pkg))
                .or_insert_with(|| vec![pkg]);
        }
        //medium frequency, low size
        else if (freq < med_freq_high_limit)
            && (freq > med_freq_low_limit)
            && (size <= med_size_low_limit)
        {
            partitions
                .entry("mf_ls".to_string())
                .and_modify(|bin| bin.push(pkg))
                .or_insert_with(|| vec![pkg]);
        }
        //high frequency, low size
        else if (freq >= med_freq_high_limit) && (size <= med_size_low_limit) {
            partitions
                .entry("hf_ls".to_string())
                .and_modify(|bin| bin.push(pkg))
                .or_insert_with(|| vec![pkg]);
        }
    }

    for (name, pkgs) in &partitions {
        println!("{:#?}: {:#?}", name, pkgs.len());
    }

    Some(partitions)
}

/// Given a set of components with size metadata (e.g. boxes of a certain size)
/// and a number of bins (possible container layers) to use, determine which components
/// go in which bin.  This algorithm is pretty simple:

// Total available bins = n
//
// 1 bin for all the u32_max frequency pkgs
// 1 bin for all newly added pkgs
// 1 bin for all low size pkgs
//
// 60% of n-3 bins for high size pkgs
// 40% of n-3 bins for medium size pkgs
//
// If HS bins > limit, spillover to MS to package
// If MS bins > limit, fold by merging 2 bins from the end
//
fn basic_packing<'a>(
    components: &'a [ObjectSourceMetaSized],
    bin_size: NonZeroU32,
    prior_build_metadata: &'a Option<Vec<Vec<String>>>,
) -> Vec<Vec<&'a ObjectSourceMetaSized>> {
    let mut r = Vec::new();
    let mut components: Vec<_> = components.iter().collect();
    let before_processing_pkgs_len = components.len();
    if before_processing_pkgs_len == 0 {
        return Vec::new();
    }
    //Flatten out prior_build_metadata[i] to view all the packages in prior build as a single vec
    //
    //If the current rpm-ostree commit to be encapsulated is not the one in which packing structure changes, then
    //  Compare flatten(prior_build_metadata[i]) to components to see if pkgs added, updated,
    //  removed or kept same
    //  if pkgs added, then add them to the last bin of prior[i][n]
    //  if pkgs removed, then remove them from the prior[i]
    //  iterate through prior[i] and make bins according to the name in nevra of pkgs and return
    //  (no need of recomputing packaging structure)
    //else if pkg structure to be changed || prior build not specified
    //  Recompute optimal packaging strcuture (Compute partitions, place packages and optimize build)

    if let Some(prior_build) = prior_build_metadata
    /* && structure not be changed*/
    {
        println!("Keeping old package structure");
        let mut curr_build: Vec<Vec<String>> = prior_build.clone();
        //Packing only manaages RPMs not OStree commit
        curr_build.remove(0);
        let mut prev_pkgs: Vec<String> = Vec::new();
        for bin in &curr_build {
            for pkg in bin {
                prev_pkgs.push(pkg.to_string());
            }
        }
        prev_pkgs.retain(|name| !name.is_empty());
        let curr_pkgs: Vec<String> = components
            .iter()
            .map(|pkg| pkg.meta.name.to_string())
            .collect();
        let prev_pkgs_set: HashSet<String> = HashSet::from_iter(prev_pkgs);
        let curr_pkgs_set: HashSet<String> = HashSet::from_iter(curr_pkgs);
        let added: HashSet<&String> = curr_pkgs_set.difference(&prev_pkgs_set).collect();
        let removed: HashSet<&String> = prev_pkgs_set.difference(&curr_pkgs_set).collect();
        let mut add_pkgs_v: Vec<String> = Vec::new();
        for pkg in added {
            add_pkgs_v.push(pkg.to_string());
        }
        let mut rem_pkgs_v: Vec<String> = Vec::new();
        for pkg in removed {
            rem_pkgs_v.push(pkg.to_string());
        }
        let curr_build_len = &curr_build.len();
        curr_build[curr_build_len - 1].retain(|name| !name.is_empty());
        curr_build[curr_build_len - 1].extend(add_pkgs_v);
        for bin in curr_build.iter_mut() {
            bin.retain(|pkg| !rem_pkgs_v.contains(pkg));
        }
        let mut name_to_component: HashMap<String, &ObjectSourceMetaSized> = HashMap::new();
        for component in &components {
            name_to_component
                .entry(component.meta.name.to_string())
                .or_insert(component);
        }
        let mut modified_build: Vec<Vec<&ObjectSourceMetaSized>> = Vec::new();
        for bin in curr_build {
            let mut mod_bin = Vec::new();
            for pkg in bin {
                mod_bin.push(name_to_component[&pkg]);
            }
            modified_build.push(mod_bin);
        }
        let mut after_processing_pkgs_len = 0;
        modified_build.iter().for_each(|bin| {
            after_processing_pkgs_len += bin.len();
        });
        assert_eq!(after_processing_pkgs_len, before_processing_pkgs_len);
        assert!(modified_build.len() <= bin_size.get() as usize);
        return modified_build;
    }

    println!("Creating new packing structure");

    let mut max_freq_components: Vec<&ObjectSourceMetaSized> = Vec::new();
    components.retain(|pkg| {
        let retain: bool = pkg.meta.change_frequency != u32::MAX;
        if !retain {
            max_freq_components.push(pkg);
        }
        retain
    });
    let components_len_after_max_freq = components.len();
    match components_len_after_max_freq {
        0 => (),
        _ => {
            //Defining Limits of each bins
            let limit_ls_bins = 1usize;
            let limit_new_bins = 1usize;
            let _limit_new_pkgs = 0usize;
            let limit_max_frequency_bins = 1usize;
            let _limit_max_frequency_pkgs = max_freq_components.len();
            let limit_hs_bins = (0.6
                * (bin_size.get()
                    - (limit_ls_bins + limit_new_bins + limit_max_frequency_bins) as u32)
                    as f32)
                .floor() as usize;
            let limit_ms_bins = (0.4
                * (bin_size.get()
                    - (limit_ls_bins + limit_new_bins + limit_max_frequency_bins) as u32)
                    as f32)
                .floor() as usize;

            let partitions =
                get_partitions_with_threshold(components, limit_hs_bins as usize, 2f64)
                    .expect("Partitioning components into sets");

            let limit_ls_pkgs = match partitions.get("2ls") {
                Some(n) => n.len(),
                None => 0usize,
            };

            let pkg_per_bin_ms: usize =
                match (components_len_after_max_freq - limit_hs_bins - limit_ls_pkgs)
                    .checked_div(limit_ms_bins)
                {
                    Some(n) => {
                        if n < 1 {
                            panic!("Error: No of bins <= 3");
                        }
                        n
                    }
                    None => {
                        panic!("Error: No of bins <= 3")
                    }
                };

            //Bins assignment
            for partition in partitions.keys() {
                let pkgs = partitions.get(partition).expect("hashset");

                if partition == "1hs" {
                    for pkg in pkgs {
                        r.push(vec![*pkg]);
                    }
                } else if partition == "2ls" {
                    let mut bin: Vec<&ObjectSourceMetaSized> = Vec::new();
                    for pkg in pkgs {
                        bin.push(*pkg);
                    }
                    r.push(bin);
                } else {
                    let mut bin: Vec<&ObjectSourceMetaSized> = Vec::new();
                    for (i, pkg) in pkgs.iter().enumerate() {
                        if bin.len() < pkg_per_bin_ms {
                            bin.push(*pkg);
                        } else {
                            r.push(bin.clone());
                            bin.clear();
                            bin.push(*pkg);
                        }
                        if i == pkgs.len() - 1 && !bin.is_empty() {
                            r.push(bin.clone());
                            bin.clear();
                        }
                    }
                }
            }
            println!("Bins before unoptimized build: {}", r.len());

            //Addressing MS bins limit breach by wrapping MS layers
            while r.len() > (bin_size.get() as usize - limit_new_bins - limit_max_frequency_bins) {
                for i in (limit_ls_bins + limit_hs_bins..r.len() - 1)
                    .step_by(2)
                    .rev()
                {
                    if r.len()
                        <= (bin_size.get() as usize - limit_new_bins - limit_max_frequency_bins)
                    {
                        break;
                    }
                    let prev = &r[i - 1];
                    let curr = &r[i];
                    let mut merge: Vec<&ObjectSourceMetaSized> = Vec::new();
                    merge.extend(prev.iter());
                    merge.extend(curr.iter());
                    r.remove(i);
                    r.remove(i - 1);
                    r.insert(i, merge);
                }
            }
            println!("Bins after optimization: {}", r.len());
        }
    }
    r.push(max_freq_components);

    let new_pkgs_bin: Vec<&ObjectSourceMetaSized> = Vec::new();
    r.push(new_pkgs_bin);
    let mut after_processing_pkgs_len = 0;
    r.iter().for_each(|bin| {
        after_processing_pkgs_len += bin.len();
    });
    assert_eq!(after_processing_pkgs_len, before_processing_pkgs_len);
    assert!(r.len() <= bin_size.get() as usize);
    r
}

#[cfg(test)]
mod test {
    use super::*;

    const FCOS_CONTENTMETA: &[u8] = include_bytes!("fixtures/fedora-coreos-contentmeta.json.gz");

    #[test]
    fn test_packing_basics() -> Result<()> {
        // null cases
        for v in [1u32, 7].map(|v| NonZeroU32::new(v).unwrap()) {
            assert_eq!(basic_packing(&[], v, &None).len(), 0);
        }
        Ok(())
    }

    #[test]
    fn test_packing_fcos() -> Result<()> {
        let contentmeta: Vec<ObjectSourceMetaSized> =
            serde_json::from_reader(flate2::read::GzDecoder::new(FCOS_CONTENTMETA))?;
        let total_size = contentmeta.iter().map(|v| v.size).sum::<u64>();

        let packing = basic_packing(&contentmeta, NonZeroU32::new(MAX_CHUNKS).unwrap(), &None);
        assert!(!contentmeta.is_empty());
        // We should fit into the assigned chunk size
        assert_eq!(packing.len() as u32, MAX_CHUNKS);
        // And verify that the sizes match
        let packed_total_size = packing_size(&packing);
        assert_eq!(total_size, packed_total_size);
        Ok(())
    }
}
