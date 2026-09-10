from unittest.mock import patch

import numpy as np
import tifffile

from negpy.infrastructure.loaders.helpers import dng_quick_preview
from negpy.infrastructure.storage.local_asset_store import LocalAssetStore
from negpy.services.assets.thumbnails import decode_source_image, get_thumbnail_worker, thumbnail_cache_key


def test_reduced_dng_ifd_is_used_without_decoding_main_pixels(tmp_path):
    path = str(tmp_path / "pyramid.dng")
    preview = np.zeros((24, 32, 3), dtype=np.uint8)
    preview[..., 1] = 180
    main = np.zeros((240, 320, 3), dtype=np.uint16)
    main[..., 0] = 65535
    with tifffile.TiffWriter(path) as writer:
        writer.write(preview, photometric="rgb", subfiletype=1, subifds=1)
        writer.write(main, photometric=34892, subfiletype=0)

    original_asarray = tifffile.TiffPage.asarray

    def reduced_only(page, *args, **kwargs):
        tag = page.tags.get("NewSubfileType")
        if tag is None or not (int(tag.value) & 1):
            raise AssertionError("full-resolution DNG pixels were decoded")
        return original_asarray(page, *args, **kwargs)

    with patch.object(tifffile.TiffPage, "asarray", reduced_only):
        result = dng_quick_preview(path)

    assert result is not None
    arr = np.asarray(result)
    assert arr.shape == preview.shape
    assert arr[..., 1].min() == 180
    assert arr[..., 0].max() == 0


def test_full_resolution_only_dng_is_not_decoded_for_a_thumbnail(tmp_path):
    path = str(tmp_path / "main-only.dng")
    tifffile.imwrite(path, np.zeros((40, 60, 3), dtype=np.uint16), photometric=34892)

    with patch("negpy.services.assets.thumbnails.loader_factory.get_loader") as loader:
        result = decode_source_image(path, quick_only=True)

    assert result is None
    loader.assert_not_called()


def test_missing_quick_preview_is_remembered(tmp_path):
    path = str(tmp_path / "main-only.dng")
    tifffile.imwrite(path, np.zeros((40, 60, 3), dtype=np.uint16), photometric=34892)
    store = LocalAssetStore(str(tmp_path / "cache"), str(tmp_path / "icc"))
    store.initialize()

    with patch("negpy.services.assets.thumbnails.decode_source_image", return_value=None) as decode:
        assert get_thumbnail_worker(path, "hash", store) is None
        assert get_thumbnail_worker(path, "hash", store) is None

    assert decode.call_count == 1
    assert store.has_thumbnail_miss(thumbnail_cache_key("hash", False))
