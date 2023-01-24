//! APIs for creating container images from OSTree commits

use super::ocidir::{Layer, OciDir};
use super::{ocidir, OstreeImageReference, Transport};
use super::{ImageReference, SignatureSource, OSTREE_COMMIT_LABEL};
use crate::chunking::{Chunk, Chunking, ObjectMetaSized};
use crate::container::skopeo;
use crate::tar as ostree_tar;
use anyhow::{anyhow, Context, Result};
use cap_std::fs::Dir;
use cap_std_ext::cap_std;
use flate2::Compression;
use fn_error_context::context;
use gio::glib;
use oci_spec::image as oci_image;
use ostree::gio;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroU32;
use std::path::Path;
use tracing::instrument;

/// The label which may be used in addition to the standard OCI label.
pub const LEGACY_VERSION_LABEL: &str = "version";

/// Type of container image generated
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ExportLayout {
    /// Actually the second layout now, but the true first one can be parsed as either
    V0,
    /// The hopefully final (optionally chunked) container image layout
    V1,
}

impl Default for ExportLayout {
    fn default() -> Self {
        Self::V1
    }
}

impl ExportLayout {
    pub(crate) fn label(&self) -> &'static str {
        match self {
            ExportLayout::V0 => "ostree.diffid",
            ExportLayout::V1 => "ostree.final-diffid",
        }
    }
}

/// Annotation injected into the layer to say that this is an ostree commit.
/// However, because this gets lost when converted to D2S2 https://docs.docker.com/registry/spec/manifest-v2-2/
/// schema, it's not actually useful today.  But, we keep it
/// out of principle.
const BLOB_OSTREE_ANNOTATION: &str = "ostree.encapsulated";
/// Configuration for the generated container.
#[derive(Debug, Default)]
pub struct Config {
    /// Additional labels.
    pub labels: Option<BTreeMap<String, String>>,
    /// The equivalent of a `Dockerfile`'s `CMD` instruction.
    pub cmd: Option<Vec<String>>,
}

fn commit_meta_to_labels<'a>(
    meta: &glib::VariantDict,
    keys: impl IntoIterator<Item = &'a str>,
    opt_keys: impl IntoIterator<Item = &'a str>,
    labels: &mut HashMap<String, String>,
) -> Result<()> {
    for k in keys {
        let v = meta
            .lookup::<String>(k)
            .context("Expected string for commit metadata value")?
            .ok_or_else(|| anyhow!("Could not find commit metadata key: {}", k))?;
        labels.insert(k.to_string(), v);
    }
    for k in opt_keys {
        let v = meta
            .lookup::<String>(k)
            .context("Expected string for commit metadata value")?;
        if let Some(v) = v {
            labels.insert(k.to_string(), v);
        }
    }
    // Copy standard metadata keys `ostree.bootable` and `ostree.linux`.
    // Bootable is an odd one out in being a boolean.
    if let Some(v) = meta.lookup::<bool>(*ostree::METADATA_KEY_BOOTABLE)? {
        labels.insert(ostree::METADATA_KEY_BOOTABLE.to_string(), v.to_string());
    }
    // Handle any other string-typed values here.
    for k in &[&ostree::METADATA_KEY_LINUX] {
        if let Some(v) = meta.lookup::<String>(k)? {
            labels.insert(k.to_string(), v);
        }
    }
    Ok(())
}

fn export_chunks(
    repo: &ostree::Repo,
    commit: &str,
    ociw: &mut OciDir,
    chunks: Vec<Chunk>,
    opts: &ExportOpts,
) -> Result<Vec<(Layer, String, Vec<String>)>> {
    chunks
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| -> Result<_> {
            let mut w = ociw.create_layer(Some(opts.compression()))?;
            ostree_tar::export_chunk(repo, commit, chunk.content, &mut w)
                .with_context(|| format!("Exporting chunk {i}"))?;
            let w = w.into_inner()?;
            Ok((w.complete()?, chunk.name, chunk.packages))
        })
        .collect()
}

/// Write an ostree commit to an OCI blob
#[context("Writing ostree root to blob")]
#[allow(clippy::too_many_arguments)]
fn export_chunked(
    repo: &ostree::Repo,
    commit: &str,
    ociw: &mut OciDir,
    manifest: &mut oci_image::ImageManifest,
    imgcfg: &mut oci_image::ImageConfiguration,
    labels: &mut HashMap<String, String>,
    mut chunking: Chunking,
    opts: &ExportOpts,
    description: &str,
) -> Result<()> {
    let layers = export_chunks(repo, commit, ociw, chunking.take_chunks(), opts)?;
    let compression = Some(opts.compression());

    match opts.format {
        ExportLayout::V0 => {
            let label = opts.format.label();
            anyhow::bail!("This legacy format using the {label} label is no longer supported");
        }
        ExportLayout::V1 => {
            // In V1, the ostree layer comes first
            let mut w = ociw.create_layer(compression)?;
            ostree_tar::export_final_chunk(repo, commit, chunking.remainder, &mut w)?;
            let w = w.into_inner()?;
            let ostree_layer = w.complete()?;

            // Then, we have a label that points to the last chunk.
            // Note in the pathological case of a single layer chunked v1 image, this could be the ostree layer.
            let last_digest = layers
                .last()
                .map(|v| &v.0)
                .unwrap_or(&ostree_layer)
                .uncompressed_sha256
                .clone();

            // Add the ostree layer
            let mut annotation_ostree_layer = HashMap::new();
            annotation_ostree_layer.insert("Content".to_string(), "ostree_commit".to_string());
            ociw.push_layer(
                manifest,
                imgcfg,
                ostree_layer,
                description,
                Some(annotation_ostree_layer),
            );
            // Add the component/content layers
            for (layer, name, packages) in layers {
                let mut annotation_component_layer = HashMap::new();
                annotation_component_layer.insert("Content".to_string(), packages.join(","));
                ociw.push_layer(
                    manifest,
                    imgcfg,
                    layer,
                    name.as_str(),
                    Some(annotation_component_layer),
                );
            }
            // This label (mentioned above) points to the last layer that is part of
            // the ostree commit.
            labels.insert(
                opts.format.label().into(),
                format!("sha256:{}", last_digest),
            );
            Ok(())
        }
    }
}

/// Generate an OCI image from a given ostree root
#[context("Building oci")]
fn build_oci(
    repo: &ostree::Repo,
    rev: &str,
    ocidir_path: &Path,
    tag: Option<&str>,
    config: &Config,
    opts: ExportOpts,
    contentmeta: Option<crate::chunking::ObjectMetaSized>,
) -> Result<ImageReference> {
    if !ocidir_path.exists() {
        std::fs::create_dir(ocidir_path).context("Creating OCI dir")?;
    }
    let ocidir = Dir::open_ambient_dir(ocidir_path, cap_std::ambient_authority())?;
    let mut writer = ocidir::OciDir::create(&ocidir)?;

    let commit = repo.require_rev(rev)?;
    let commit = commit.as_str();
    let (commit_v, _) = repo.load_commit(commit)?;
    let commit_subject = commit_v.child_value(3);
    let commit_subject = commit_subject.str().ok_or_else(|| {
        anyhow::anyhow!(
            "Corrupted commit {}; expecting string value for subject",
            commit
        )
    })?;
    let commit_meta = &commit_v.child_value(0);
    let commit_meta = glib::VariantDict::new(Some(commit_meta));

    let mut ctrcfg = oci_image::Config::default();
    let mut imgcfg = oci_image::ImageConfiguration::default();
    let labels = ctrcfg.labels_mut().get_or_insert_with(Default::default);

    commit_meta_to_labels(
        &commit_meta,
        opts.copy_meta_keys.iter().map(|k| k.as_str()),
        opts.copy_meta_opt_keys.iter().map(|k| k.as_str()),
        labels,
    )?;

    let mut manifest = ocidir::new_empty_manifest().build().unwrap();

    let chunking = contentmeta
        .map(|meta| {
            crate::chunking::Chunking::from_mapping(
                repo,
                commit,
                meta,
                &opts.max_layers,
                &opts.prior_build_metadata,
            )
        })
        .transpose()?;
    // If no chunking was provided, create a logical single chunk.
    let chunking = chunking
        .map(Ok)
        .unwrap_or_else(|| crate::chunking::Chunking::new(repo, commit))?;

    if let Some(version) = commit_meta.lookup::<String>("version")? {
        if !opts.no_legacy_version_label {
            labels.insert(LEGACY_VERSION_LABEL.into(), version.clone());
        }
        labels.insert(oci_image::ANNOTATION_VERSION.into(), version);
    }
    labels.insert(OSTREE_COMMIT_LABEL.into(), commit.into());

    for (k, v) in config.labels.iter().flat_map(|k| k.iter()) {
        labels.insert(k.into(), v.into());
    }

    let mut annos = HashMap::new();
    annos.insert(BLOB_OSTREE_ANNOTATION.to_string(), "true".to_string());
    let description = if commit_subject.is_empty() {
        Cow::Owned(format!("ostree export of commit {}", commit))
    } else {
        Cow::Borrowed(commit_subject)
    };

    export_chunked(
        repo,
        commit,
        &mut writer,
        &mut manifest,
        &mut imgcfg,
        labels,
        chunking,
        &opts,
        &description,
    )?;

    // Lookup the cmd embedded in commit metadata
    let cmd = commit_meta.lookup::<Vec<String>>(ostree::COMMIT_META_CONTAINER_CMD)?;
    // But support it being overridden by CLI options

    // https://github.com/rust-lang/rust-clippy/pull/7639#issuecomment-1050340564
    #[allow(clippy::unnecessary_lazy_evaluations)]
    let cmd = config.cmd.as_ref().or_else(|| cmd.as_ref());
    if let Some(cmd) = cmd {
        ctrcfg.set_cmd(Some(cmd.clone()));
    }

    imgcfg.set_config(Some(ctrcfg));
    let ctrcfg = writer.write_config(imgcfg)?;
    manifest.set_config(ctrcfg);
    let platform = oci_image::Platform::default();
    if let Some(tag) = tag {
        writer.insert_manifest(manifest, Some(tag), platform)?;
    } else {
        writer.replace_with_single_manifest(manifest, platform)?;
    }

    Ok(ImageReference {
        transport: Transport::OciDir,
        name: ocidir_path.to_str().unwrap().to_string(),
    })
}

/// Interpret a filesystem path as optionally including a tag.  Paths
/// such as `/foo/bar` will return `("/foo/bar"`, None)`, whereas
/// e.g. `/foo/bar:latest` will return `("/foo/bar", Some("latest"))`.
pub(crate) fn parse_oci_path_and_tag(path: &str) -> (&str, Option<&str>) {
    match path.rsplit_once(':') {
        Some((path, tag)) => (path, Some(tag)),
        None => (path, None),
    }
}

/// Helper for `build()` that avoids generics
#[instrument(skip(repo, contentmeta))]
async fn build_impl(
    repo: &ostree::Repo,
    ostree_ref: &str,
    config: &Config,
    opts: Option<ExportOpts>,
    contentmeta: Option<ObjectMetaSized>,
    dest: &ImageReference,
) -> Result<String> {
    let mut opts = opts.unwrap_or_default();
    if dest.transport == Transport::ContainerStorage {
        opts.skip_compression = true;
    }
    let digest = if dest.transport == Transport::OciDir {
        let (path, tag) = parse_oci_path_and_tag(dest.name.as_str());
        let _copied: ImageReference = build_oci(
            repo,
            ostree_ref,
            Path::new(path),
            tag,
            config,
            opts,
            contentmeta,
        )?;
        None
    } else {
        let tempdir = tempfile::tempdir_in("/var/tmp")?;
        let tempdest = tempdir.path().join("d");
        let tempdest = tempdest.to_str().unwrap();

        let tempoci = build_oci(
            repo,
            ostree_ref,
            Path::new(tempdest),
            None,
            config,
            opts,
            contentmeta,
        )?;

        let digest = skopeo::copy(&tempoci, dest).await?;
        Some(digest)
    };
    if let Some(digest) = digest {
        Ok(digest)
    } else {
        // If `skopeo copy` doesn't have `--digestfile` yet, then fall back
        // to running an inspect cycle.
        let imgref = OstreeImageReference {
            sigverify: SignatureSource::ContainerPolicyAllowInsecure,
            imgref: dest.to_owned(),
        };
        let (_, digest) = super::unencapsulate::fetch_manifest(&imgref).await?;
        Ok(digest)
    }
}

/// Options controlling commit export into OCI
#[derive(Debug, Default)]
pub struct ExportOpts {
    /// If true, do not perform gzip compression of the tar layers.
    pub skip_compression: bool,
    /// A set of commit metadata keys to copy as image labels.
    pub copy_meta_keys: Vec<String>,
    /// A set of optionally-present commit metadata keys to copy as image labels.
    pub copy_meta_opt_keys: Vec<String>,
    /// Maximum number of layers to use
    pub max_layers: Option<NonZeroU32>,
    /// The container image layout
    pub format: ExportLayout,
    // TODO semver-break: remove this
    /// Use only the standard OCI version label
    pub no_legacy_version_label: bool,
    /// Prevent major change in packaging structure by taking previous builds in order of priority
    pub prior_build_metadata: Option<Vec<Vec<String>>>,
}

impl ExportOpts {
    /// Return the gzip compression level to use, as configured by the export options.
    fn compression(&self) -> Compression {
        if self.skip_compression {
            Compression::fast()
        } else {
            Compression::default()
        }
    }
}

/// Given an OSTree repository and ref, generate a container image.
///
/// The returned `ImageReference` will contain a digested (e.g. `@sha256:`) version of the destination.
pub async fn encapsulate<S: AsRef<str>>(
    repo: &ostree::Repo,
    ostree_ref: S,
    config: &Config,
    opts: Option<ExportOpts>,
    contentmeta: Option<ObjectMetaSized>,
    dest: &ImageReference,
) -> Result<String> {
    build_impl(repo, ostree_ref.as_ref(), config, opts, contentmeta, dest).await
}
