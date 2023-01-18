//! Split an OSTree commit into separate chunks

// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::borrow::{Borrow, Cow};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write;
use std::hash::{Hash, Hasher};
use std::num::NonZeroU32;
use std::rc::Rc;

use crate::objectsource::{ContentID, ObjectMeta, ObjectMetaMap, ObjectSourceMeta};
use crate::objgv::*;
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
    pub(crate) packages: Vec<String>
}

#[derive(Debug, Deserialize, Serialize, Clone)]
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
        prior_build_metadata: &Option<Vec<Vec<String>>>
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
        prior_build_metadata: &Option<Vec<Vec<String>>>
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
        let packing = basic_packing(sizes, NonZeroU32::new(self.max).unwrap(), prior_build_metadata);

        for bin in packing.into_iter() {
            let first = bin[0];
            let first_name = &*first.meta.name;
            let name = match bin.len() {
                0 => unreachable!(),
                1 => Cow::Borrowed(first_name),
                2..=5 => {
                    let r = bin.iter().map(|v| &*v.meta.name).fold(
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
            if !chunk.content.is_empty() {
                self.chunks.push(chunk);
            }
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

type ChunkedComponents<'a> = Vec<&'a ObjectSourceMetaSized>;

fn components_size(components: &[&ObjectSourceMetaSized]) -> u64 {
    components.iter().map(|k| k.size).sum()
}

/// Compute the total size of a packing
#[cfg(test)]
fn packing_size(packing: &[ChunkedComponents]) -> u64 {
    packing.iter().map(|v| components_size(v)).sum()
}

fn sort_packing(packing: &mut [ChunkedComponents]) {
    packing.sort_by(|a, b| {
        let a: u64 = components_size(a);
        let b: u64 = components_size(b);
        b.cmp(&a)
    });
}

fn mean(data: &[u64]) -> Option<f64> {
    let sum = data.iter().sum::<u64>() as f64;
    let count = data.len();
    match count {
        positive if positive > 0 => Some(sum / count as f64),
         _ => None,
    }
}

fn std_deviation(data: &[u64]) -> Option<f64> {
    match (mean(data), data.len()) {
        (Some(data_mean), count) if count > 0 => {
            let variance = data.iter().map(|value| {
                let diff = data_mean - (*value as f64);
                diff * diff
            }).sum::<f64>() / count as f64;
            Some(variance.sqrt())
        },
        _ => None
    }
}

fn get_partitions_with_threshold(components: Vec<&ObjectSourceMetaSized>, threshold: f64) -> Option<HashMap<String, Vec<&ObjectSourceMetaSized>>>{
    let frequencies: Vec<u64> = components.iter().map(|a| a.meta.change_frequency.into()).collect();
    let sizes: Vec<u64> = components.iter().map(|a| a.size).collect();
    let mean_freq = mean(&frequencies)?;
    let stddev_freq = std_deviation(&frequencies)?;
    let mean_size = mean(&sizes)?;
    let stddev_size = std_deviation(&sizes)?;
    let mut bins : HashMap<String, Vec<&ObjectSourceMetaSized>> = HashMap::new();

    let mut freq_low_limit = mean_freq - threshold*stddev_freq;
    if freq_low_limit < 0 as f64 {
        freq_low_limit = 1 as f64;
    }
    let freq_high_limit = mean_freq + threshold*stddev_freq;
    let mut size_low_limit = mean_size - threshold*stddev_size;
    if size_low_limit < 0 as f64 {
        size_low_limit = 100000 as f64;
    }
    let size_high_limit = mean_size + threshold*stddev_size;
    
    for pkg in components {
        let size = pkg.size as f64;
        let freq = pkg.meta.change_frequency as f64;
        
        //lf_hs
        if  (freq <= freq_low_limit) && 
            (size >= size_high_limit) {
            bins.entry("lf_hs".to_string()).and_modify(|bin| bin.push(pkg)).or_insert(vec!(pkg));
        }

        //mf_hs
        else if (freq < freq_high_limit) && (freq > freq_low_limit) && 
                (size >= size_high_limit) {
            bins.entry("mf_hs".to_string()).and_modify(|bin| bin.push(pkg)).or_insert(vec!(pkg));
        }

        //hf_hs
        else if (freq >= freq_high_limit) && 
                (size >= size_high_limit) {
            bins.entry("hf_hs".to_string()).and_modify(|bin| bin.push(pkg)).or_insert(vec!(pkg));
        }

        //lf_ms
        else if (freq <= freq_low_limit) && 
                (size < size_high_limit) && (size > size_low_limit) {
            bins.entry("lf_ms".to_string()).and_modify(|bin| bin.push(pkg)).or_insert(vec!(pkg));
        }

        //mf_ms
        else if (freq < freq_high_limit) && (freq > freq_low_limit) && 
                (size < size_high_limit) && (size > size_low_limit){
            bins.entry("mf_ms".to_string()).and_modify(|bin| bin.push(pkg)).or_insert(vec!(pkg));
        }

        //hf_ms
        else if (freq >= freq_high_limit) && 
                (size < size_high_limit) && (size > size_low_limit) {
            bins.entry("hf_ms".to_string()).and_modify(|bin| bin.push(pkg)).or_insert(vec!(pkg));
        }
        
        //lf_ls
        else if (freq <= freq_low_limit) && 
                (size <= size_low_limit) {
            bins.entry("lf_ls".to_string()).and_modify(|bin| bin.push(pkg)).or_insert(vec!(pkg));
        }

        //mf_ls
        else if (freq < freq_high_limit) && (freq > freq_low_limit) && 
                (size <= size_low_limit) {
            bins.entry("mf_ls".to_string()).and_modify(|bin| bin.push(pkg)).or_insert(vec!(pkg));
        }

        //hf_ls
        else if (freq >= freq_high_limit) && 
                (size <= size_low_limit) {
            bins.entry("hf_ls".to_string()).and_modify(|bin| bin.push(pkg)).or_insert(vec!(pkg));
        }
    }

    for (name, pkgs) in &bins {
        println!("{:#?}: {:#?}", name, pkgs.len());
    }

    Some(bins)
}

/// Given a set of components with size metadata (e.g. boxes of a certain size)
/// and a number of bins (possible container layers) to use, determine which components
/// go in which bin.  This algorithm is pretty simple:
///
fn basic_packing<'a>(components: &'a [ObjectSourceMetaSized], bin_size: NonZeroU32, prior_builds_metadata: &'a Option<Vec<Vec<Vec<String>>>>) -> Vec<ChunkedComponents<'a>> {
    let mut r = Vec::new();
    let mut components: Vec<_> = components.iter().collect();
    let before_processing_pkgs_len = components.len();
    components.sort_by(|a, b| a.meta.change_frequency.cmp(&b.meta.change_frequency));
    let mut max_freq_components: Vec<&ObjectSourceMetaSized> = Vec::new();
    components.retain(|pkg| {
        let retain: bool = pkg.meta.change_frequency != u32::MAX;
        if !retain {
            max_freq_components.push(pkg);
        } 
        retain
    });
    let max_freq_len = max_freq_components.len();
    let partitions = get_partitions_with_threshold(components, 1.5).expect("Partitioning components into sets");
    for pkgs in partitions.values() {
        let max_bin_size: u64 = pkgs.iter().map(|a| a.size).max().unwrap();
        let mut bin_size = 0;
        let mut bin : Vec<&ObjectSourceMetaSized> = Vec::new();
        //Index of pkg in pkgs where the bin begins 
        let mut bin_start_index = 0;
        for (i, pkg) in pkgs.iter().enumerate(){
            let size_pkg = pkg.size;
            bin_size += size_pkg;
            bin.push(pkg); 

            if bin_size > max_bin_size {
                bin.pop();
                r.push(bin.clone());
                bin.clear();
                bin.push(pkg);
                bin_size = pkg.size;
                bin_start_index = i;
            }

            else if bin_size == max_bin_size {
                r.push(bin.clone());
                bin_size = 0;
                bin.clear();
                bin_start_index = i + 1; 
            }

            if i == pkgs.len() - 1 && bin.len() != 0 {
                r.push(bin.clone());
                bin.clear();
            }
        }
    }

    println!("Bins before unoptimized build: {}", r.len());
    while r.len() > (bin_size.get() - 1) as usize {
        for i in (1..r.len() - 1).step_by(2).rev() {
            if r.len() <= (bin_size.get() - 1) as usize {
                break;          
            }
            let prev = &r[i-1];
            let curr = &r[i];
            let mut merge: Vec<&ObjectSourceMetaSized> = Vec::new();
            merge.extend(prev.into_iter());
            merge.extend(curr.into_iter());
            r.remove(i);
            r.remove(i-1);
            r.insert(i, merge);
        }
    }
    println!("Bins after optimization: {}", r.len());
    r.push(max_freq_components);
    let mut after_processing_pkgs_len = 0;
    r.iter().for_each(|bin| {
        after_processing_pkgs_len += bin.len();
    });
    assert!(after_processing_pkgs_len == before_processing_pkgs_len);
    //Use the previous builds data to calculate difference in packaging. If more than 70% changed,
    //shift packing structure otherwise make changes to adhere to it
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
            assert_eq!(basic_packing(&[], v).len(), 0);
        }
        Ok(())
    }

    #[test]
    fn test_packing_fcos() -> Result<()> {
        let contentmeta: Vec<ObjectSourceMetaSized> =
            serde_json::from_reader(flate2::read::GzDecoder::new(FCOS_CONTENTMETA))?;
        let total_size = contentmeta.iter().map(|v| v.size).sum::<u64>();

        let packing = basic_packing(&contentmeta, NonZeroU32::new(MAX_CHUNKS).unwrap());
        assert!(!contentmeta.is_empty());
        // We should fit into the assigned chunk size
        assert_eq!(packing.len() as u32, MAX_CHUNKS);
        // And verify that the sizes match
        let packed_total_size = packing_size(&packing);
        assert_eq!(total_size, packed_total_size);
        Ok(())
    }
}
