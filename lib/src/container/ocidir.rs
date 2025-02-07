//! Internal API to interact with Open Container Images; mostly
//! oriented towards generating images.
//!
//! NOTE: Everything in here is `pub`, but that's only used to
//! expose this API when we're running our own tests.

use anyhow::{anyhow, Context, Result};
use camino::Utf8Path;
use cap_std::fs::Dir;
use cap_std_ext::cap_std;
use cap_std_ext::dirext::CapStdExtDirExt;
use containers_image_proxy::oci_spec;
use flate2::write::GzEncoder;
use fn_error_context::context;
use oci_image::MediaType;
use oci_spec::image::{self as oci_image, Descriptor};
use olpc_cjson::CanonicalFormatter;
use openssl::hash::{Hasher, MessageDigest};
use serde::Serialize;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::File;
use std::io::{prelude::*, BufReader};
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};

/// Path inside an OCI directory to the blobs
const BLOBDIR: &str = "blobs/sha256";

const OCI_TAG_ANNOTATION: &str = "org.opencontainers.image.ref.name";

/// Completed blob metadata
#[derive(Debug)]
pub struct Blob {
    /// SHA-256 digest
    pub sha256: String,
    /// Size
    pub size: u64,
}

impl Blob {
    /// The OCI standard checksum-type:checksum
    pub fn digest_id(&self) -> String {
        format!("sha256:{}", self.sha256)
    }

    /// Descriptor
    pub fn descriptor(&self) -> oci_image::DescriptorBuilder {
        oci_image::DescriptorBuilder::default()
            .digest(self.digest_id())
            .size(self.size as i64)
    }
}

/// Completed layer metadata
#[derive(Debug)]
pub struct Layer {
    /// The underlying blob (usually compressed)
    pub blob: Blob,
    /// The uncompressed digest, which will be used for "diffid"s
    pub uncompressed_sha256: String,
}

impl Layer {
    /// Return the descriptor for this layer
    pub fn descriptor(&self) -> oci_image::DescriptorBuilder {
        self.blob.descriptor()
    }
}

/// Create an OCI blob.
pub struct BlobWriter<'a> {
    /// Compute checksum
    pub hash: Hasher,
    /// Target file
    pub target: Option<cap_tempfile::TempFile<'a>>,
    size: u64,
}

impl<'a> Debug for BlobWriter<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlobWriter")
            .field("target", &self.target)
            .field("size", &self.size)
            .finish()
    }
}

/// Create an OCI layer (also a blob).
pub struct RawLayerWriter<'a> {
    bw: BlobWriter<'a>,
    uncompressed_hash: Hasher,
    compressor: GzEncoder<Vec<u8>>,
}

impl<'a> Debug for RawLayerWriter<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawLayerWriter")
            .field("bw", &self.bw)
            .field("compressor", &self.compressor)
            .finish()
    }
}

#[derive(Debug)]
/// An opened OCI directory.
pub struct OciDir {
    /// The underlying directory.
    pub dir: std::sync::Arc<Dir>,
}

/// Write a serializable data (JSON) as an OCI blob
#[context("Writing json blob")]
pub fn write_json_blob<S: serde::Serialize>(
    ocidir: &Dir,
    v: &S,
    media_type: oci_image::MediaType,
) -> Result<oci_image::DescriptorBuilder> {
    let mut w = BlobWriter::new(ocidir)?;
    let mut ser = serde_json::Serializer::with_formatter(&mut w, CanonicalFormatter::new());
    v.serialize(&mut ser).context("Failed to serialize")?;
    let blob = w.complete()?;
    Ok(blob.descriptor().media_type(media_type))
}

// Parse a filename from a string; this will ignore any directory components, and error out on `/` and `..` for example.
fn parse_one_filename(s: &str) -> Result<&str> {
    Utf8Path::new(s)
        .file_name()
        .ok_or_else(|| anyhow!("Invalid filename {}", s))
}

/// Create a dummy config descriptor.
/// Our API right now always mutates a manifest, which means we need
/// a "valid" manifest, which requires a "valid" config descriptor.
/// This digest should never actually be used for anything.
fn empty_config_descriptor() -> oci_image::Descriptor {
    oci_image::DescriptorBuilder::default()
        .media_type(MediaType::ImageConfig)
        .size(7023)
        .digest("sha256:a5b2b2c507a0944348e0303114d8d93aaaa081732b86451d9bce1f432a537bc7")
        .build()
        .unwrap()
}

/// Generate a "valid" empty manifest.  See above.
pub fn new_empty_manifest() -> oci_image::ImageManifestBuilder {
    oci_image::ImageManifestBuilder::default()
        .schema_version(oci_image::SCHEMA_VERSION)
        .config(empty_config_descriptor())
        .layers(Vec::new())
}

impl OciDir {
    /// Create a new, empty OCI directory at the target path, which should be empty.
    pub fn create(dir: &Dir) -> Result<Self> {
        let mut db = cap_std::fs::DirBuilder::new();
        db.recursive(true).mode(0o755);
        dir.ensure_dir_with(BLOBDIR, &db)?;
        dir.atomic_write("oci-layout", r#"{"imageLayoutVersion":"1.0.0"}"#)?;
        Self::open(dir)
    }

    /// Clone an OCI directory, using reflinks for blobs.
    pub fn clone_to(&self, destdir: &Dir, p: impl AsRef<Path>) -> Result<Self> {
        let p = p.as_ref();
        destdir.create_dir(p)?;
        let cloned = Self::create(&destdir.open_dir(p)?)?;
        for blob in self.dir.read_dir(BLOBDIR)? {
            let blob = blob?;
            let path = Path::new(BLOBDIR).join(blob.file_name());
            let mut src = self.dir.open(&path).map(BufReader::new)?;
            self.dir
                .atomic_replace_with(&path, |w| std::io::copy(&mut src, w))?;
        }
        Ok(cloned)
    }

    /// Open an existing OCI directory.
    pub fn open(dir: &Dir) -> Result<Self> {
        let dir = std::sync::Arc::new(dir.try_clone()?);
        Ok(Self { dir })
    }

    /// Create a writer for a new blob (expected to be a tar stream)
    pub fn create_raw_layer(&self, c: Option<flate2::Compression>) -> Result<RawLayerWriter> {
        RawLayerWriter::new(&self.dir, c)
    }

    /// Create a tar output stream, backed by a blob
    pub fn create_layer(
        &self,
        c: Option<flate2::Compression>,
    ) -> Result<tar::Builder<RawLayerWriter>> {
        Ok(tar::Builder::new(self.create_raw_layer(c)?))
    }

    /// Add a layer to the top of the image stack.  The firsh pushed layer becomes the root.

    pub fn push_layer(
        &self,
        manifest: &mut oci_image::ImageManifest,
        config: &mut oci_image::ImageConfiguration,
        layer: Layer,
        description: &str,
        annotations: Option<HashMap<String, String>>,
    ) {
        self.push_layer_annotated(manifest, config, layer, annotations, description);
    }

    /// Add a layer to the top of the image stack with optional annotations.
    ///
    /// This is otherwise equivalent to [`Self::push_layer`].
    pub fn push_layer_annotated(
        &self,
        manifest: &mut oci_image::ImageManifest,
        config: &mut oci_image::ImageConfiguration,
        layer: Layer,
        annotations: Option<impl Into<HashMap<String, String>>>,
        description: &str,
    ) {
        let mut builder = layer.descriptor().media_type(MediaType::ImageLayerGzip);
        if let Some(annotations) = annotations {
            builder = builder.annotations(annotations);
        }
        let blobdesc = builder.build().unwrap();
        manifest.layers_mut().push(blobdesc);
        let mut rootfs = config.rootfs().clone();
        rootfs
            .diff_ids_mut()
            .push(format!("sha256:{}", layer.uncompressed_sha256));
        config.set_rootfs(rootfs);
        let now = chrono::offset::Utc::now();
        let h = oci_image::HistoryBuilder::default()
            .created(now.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
            .created_by(description.to_string())
            .build()
            .unwrap();
        config.history_mut().push(h);
    }

    fn parse_descriptor_to_path(desc: &oci_spec::image::Descriptor) -> Result<PathBuf> {
        let (alg, hash) = desc
            .digest()
            .split_once(':')
            .ok_or_else(|| anyhow!("Invalid digest {}", desc.digest()))?;
        let alg = parse_one_filename(alg)?;
        if alg != "sha256" {
            anyhow::bail!("Unsupported digest algorithm {}", desc.digest());
        }
        let hash = parse_one_filename(hash)?;
        Ok(Path::new(BLOBDIR).join(hash))
    }

    /// Open a blob
    pub fn read_blob(&self, desc: &oci_spec::image::Descriptor) -> Result<File> {
        let path = Self::parse_descriptor_to_path(desc)?;
        self.dir
            .open(path)
            .map_err(Into::into)
            .map(|f| f.into_std())
    }

    /// Read a JSON blob.
    pub fn read_json_blob<T: serde::de::DeserializeOwned + Send + 'static>(
        &self,
        desc: &oci_spec::image::Descriptor,
    ) -> Result<T> {
        let blob = BufReader::new(self.read_blob(desc)?);
        serde_json::from_reader(blob).with_context(|| format!("Parsing object {}", desc.digest()))
    }

    /// Write a configuration blob.
    pub fn write_config(
        &self,
        config: oci_image::ImageConfiguration,
    ) -> Result<oci_image::Descriptor> {
        Ok(write_json_blob(&self.dir, &config, MediaType::ImageConfig)?
            .build()
            .unwrap())
    }

    /// Write a manifest as a blob, and replace the index with a reference to it.
    pub fn insert_manifest(
        &self,
        manifest: oci_image::ImageManifest,
        tag: Option<&str>,
        platform: oci_image::Platform,
    ) -> Result<Descriptor> {
        let mut manifest = write_json_blob(&self.dir, &manifest, MediaType::ImageManifest)?
            .platform(platform)
            .build()
            .unwrap();
        if let Some(tag) = tag {
            let annotations: HashMap<_, _> = [(OCI_TAG_ANNOTATION.to_string(), tag.to_string())]
                .into_iter()
                .collect();
            manifest.set_annotations(Some(annotations));
        }

        let index = self.dir.open_optional("index.json")?.map(BufReader::new);
        let index =
            if let Some(mut index) = index.map(oci_image::ImageIndex::from_reader).transpose()? {
                let mut manifests = index.manifests().clone();
                manifests.push(manifest.clone());
                index.set_manifests(manifests);
                index
            } else {
                oci_image::ImageIndexBuilder::default()
                    .schema_version(oci_image::SCHEMA_VERSION)
                    .manifests(vec![manifest.clone()])
                    .build()
                    .unwrap()
            };

        self.dir
            .atomic_replace_with("index.json", |mut w| -> Result<()> {
                let mut ser =
                    serde_json::Serializer::with_formatter(&mut w, CanonicalFormatter::new());
                index.serialize(&mut ser).context("Failed to serialize")?;
                Ok(())
            })?;
        Ok(manifest)
    }

    /// Write a manifest as a blob, and replace the index with a reference to it.
    pub fn replace_with_single_manifest(
        &self,
        manifest: oci_image::ImageManifest,
        platform: oci_image::Platform,
    ) -> Result<()> {
        let manifest = write_json_blob(&self.dir, &manifest, MediaType::ImageManifest)?
            .platform(platform)
            .build()
            .unwrap();

        let index_data = oci_image::ImageIndexBuilder::default()
            .schema_version(oci_image::SCHEMA_VERSION)
            .manifests(vec![manifest])
            .build()
            .unwrap();
        self.dir
            .atomic_replace_with("index.json", |mut w| -> Result<()> {
                let mut ser =
                    serde_json::Serializer::with_formatter(&mut w, CanonicalFormatter::new());
                index_data
                    .serialize(&mut ser)
                    .context("Failed to serialize")?;
                Ok(())
            })?;
        Ok(())
    }

    /// If this OCI directory has a single manifest, return it.  Otherwise, an error is returned.
    pub fn read_manifest(&self) -> Result<oci_image::ImageManifest> {
        self.read_manifest_and_descriptor().map(|r| r.0)
    }

    /// Find the manifest with the provided tag
    pub fn find_manifest_with_tag(&self, tag: &str) -> Result<Option<oci_image::ImageManifest>> {
        let f = self
            .dir
            .open("index.json")
            .context("Failed to open index.json")?;
        let idx: oci_image::ImageIndex = serde_json::from_reader(BufReader::new(f))?;
        for img in idx.manifests() {
            if img
                .annotations()
                .as_ref()
                .and_then(|annos| annos.get(OCI_TAG_ANNOTATION))
                .filter(|tagval| tagval.as_str() == tag)
                .is_some()
            {
                return self.read_json_blob(img).map(Some);
            }
        }
        Ok(None)
    }

    /// If this OCI directory has a single manifest, return it.  Otherwise, an error is returned.
    pub fn read_manifest_and_descriptor(&self) -> Result<(oci_image::ImageManifest, Descriptor)> {
        let f = self
            .dir
            .open("index.json")
            .context("Failed to open index.json")?;
        let idx: oci_image::ImageIndex = serde_json::from_reader(BufReader::new(f))?;
        let desc = match idx.manifests().as_slice() {
            [] => anyhow::bail!("No manifests found"),
            [desc] => desc.clone(),
            manifests => anyhow::bail!("Expected exactly 1 manifest, found {}", manifests.len()),
        };
        Ok((self.read_json_blob(&desc)?, desc))
    }
}

impl<'a> BlobWriter<'a> {
    #[context("Creating blob writer")]
    fn new(ocidir: &'a Dir) -> Result<Self> {
        Ok(Self {
            hash: Hasher::new(MessageDigest::sha256())?,
            // FIXME add ability to choose filename after completion
            target: Some(cap_tempfile::TempFile::new(ocidir)?),
            size: 0,
        })
    }

    #[context("Completing blob")]
    /// Finish writing this blob object.
    pub fn complete(mut self) -> Result<Blob> {
        let sha256 = hex::encode(self.hash.finish()?);
        let destname = &format!("{}/{}", BLOBDIR, sha256);
        let target = self.target.take().unwrap();
        target.replace(destname)?;
        Ok(Blob {
            sha256,
            size: self.size,
        })
    }
}

impl<'a> std::io::Write for BlobWriter<'a> {
    fn write(&mut self, srcbuf: &[u8]) -> std::io::Result<usize> {
        self.hash.update(srcbuf)?;
        self.target
            .as_mut()
            .unwrap()
            .as_file_mut()
            .write_all(srcbuf)?;
        self.size += srcbuf.len() as u64;
        Ok(srcbuf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> RawLayerWriter<'a> {
    /// Create a writer for a gzip compressed layer blob.
    fn new(ocidir: &'a Dir, c: Option<flate2::Compression>) -> Result<Self> {
        let bw = BlobWriter::new(ocidir)?;
        Ok(Self {
            bw,
            uncompressed_hash: Hasher::new(MessageDigest::sha256())?,
            compressor: GzEncoder::new(Vec::with_capacity(8192), c.unwrap_or_default()),
        })
    }

    #[context("Completing layer")]
    /// Consume this writer, flushing buffered data and put the blob in place.
    pub fn complete(mut self) -> Result<Layer> {
        self.compressor.get_mut().clear();
        let buf = self.compressor.finish()?;
        self.bw.write_all(&buf)?;
        let blob = self.bw.complete()?;
        let uncompressed_sha256 = hex::encode(self.uncompressed_hash.finish()?);
        Ok(Layer {
            blob,
            uncompressed_sha256,
        })
    }
}

impl<'a> std::io::Write for RawLayerWriter<'a> {
    fn write(&mut self, srcbuf: &[u8]) -> std::io::Result<usize> {
        self.compressor.get_mut().clear();
        self.compressor.write_all(srcbuf).unwrap();
        self.uncompressed_hash.update(srcbuf)?;
        let compressed_buf = self.compressor.get_mut().as_slice();
        self.bw.write_all(compressed_buf)?;
        Ok(srcbuf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.bw.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST_DERIVE: &str = r#"{
        "schemaVersion": 2,
        "config": {
          "mediaType": "application/vnd.oci.image.config.v1+json",
          "digest": "sha256:54977ab597b345c2238ba28fe18aad751e5c59dc38b9393f6f349255f0daa7fc",
          "size": 754
        },
        "layers": [
          {
            "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
            "digest": "sha256:ee02768e65e6fb2bb7058282338896282910f3560de3e0d6cd9b1d5985e8360d",
            "size": 5462
          },
          {
            "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
            "digest": "sha256:d203cef7e598fa167cb9e8b703f9f20f746397eca49b51491da158d64968b429",
            "size": 214
          }
        ],
        "annotations": {
          "ostree.commit": "3cb6170b6945065c2475bc16d7bebcc84f96b4c677811a6751e479b89f8c3770",
          "ostree.version": "42.0"
        }
      }
    "#;

    #[test]
    fn manifest() -> Result<()> {
        let m: oci_image::ImageManifest = serde_json::from_str(MANIFEST_DERIVE)?;
        assert_eq!(
            m.layers()[0].digest().as_str(),
            "sha256:ee02768e65e6fb2bb7058282338896282910f3560de3e0d6cd9b1d5985e8360d"
        );
        Ok(())
    }

    #[test]
    fn test_build() -> Result<()> {
        let td = cap_tempfile::tempdir(cap_std::ambient_authority())?;
        let w = OciDir::create(&td)?;
        let mut layerw = w.create_raw_layer(None)?;
        layerw.write_all(b"pretend this is a tarball")?;
        let root_layer = layerw.complete()?;
        assert_eq!(
            root_layer.uncompressed_sha256,
            "349438e5faf763e8875b43de4d7101540ef4d865190336c2cc549a11f33f8d7c"
        );
        let mut manifest = new_empty_manifest().build().unwrap();
        let mut config = oci_image::ImageConfigurationBuilder::default()
            .build()
            .unwrap();
        let annotations: Option<HashMap<String, String>> = None;
        w.push_layer(&mut manifest, &mut config, root_layer, "root", annotations);
        let config = w.write_config(config)?;
        manifest.set_config(config);
        w.replace_with_single_manifest(manifest.clone(), oci_image::Platform::default())?;

        let read_manifest = w.read_manifest().unwrap();
        assert_eq!(&read_manifest, &manifest);

        let _: Descriptor =
            w.insert_manifest(manifest, Some("latest"), oci_image::Platform::default())?;
        // There's more than one now
        assert!(w.read_manifest().is_err());
        assert!(w.find_manifest_with_tag("noent").unwrap().is_none());
        let found_via_tag = w.find_manifest_with_tag("latest").unwrap().unwrap();
        assert_eq!(found_via_tag, read_manifest);

        Ok(())
    }
}
