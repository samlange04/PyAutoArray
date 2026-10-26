import importlib.util
import os
import subprocess
import sys
import textwrap

import autoarray as aa

from autoarray.inversion.inversion.imaging.inversion_imaging_util import (
    ImagingSparseOperator,
)
from autoarray.inversion.inversion.imaging.sparse import (
    InversionImagingSparse,
)

from autoarray import exc

import numpy as np
import pytest
from pathlib import Path

directory = Path(__file__).resolve().parent


def test__operated_mapping_matrix_property(psf_3x3, rectangular_mapper_7x7_3x3):

    inversion = aa.m.MockInversionImaging(
        mask=rectangular_mapper_7x7_3x3.mask,
        psf=psf_3x3,
        linear_obj_list=[rectangular_mapper_7x7_3x3],
    )

    assert inversion.operated_mapping_matrix_list[0][0, 0] == pytest.approx(
        1.61999997, 1e-4
    )
    assert inversion.operated_mapping_matrix[0, 0] == pytest.approx(1.61999997408, 1e-4)

    mask = aa.Mask2D(
        [
            [True, True, True, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=1.0,
    )
    psf = aa.m.MockPSF(operated_mapping_matrix=np.ones((2, 2)))

    inversion = aa.m.MockInversionImaging(
        mask=mask,
        psf=psf,
        linear_obj_list=[rectangular_mapper_7x7_3x3, rectangular_mapper_7x7_3x3],
    )

    operated_mapping_matrix_0 = np.array([[1.0, 1.0], [1.0, 1.0]])
    operated_mapping_matrix_1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    operated_mapping_matrix = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

    assert inversion.operated_mapping_matrix_list[0] == pytest.approx(
        operated_mapping_matrix_0, 1.0e-4
    )
    assert inversion.operated_mapping_matrix_list[1] == pytest.approx(
        operated_mapping_matrix_1, 1.0e-4
    )
    assert inversion.operated_mapping_matrix == pytest.approx(
        operated_mapping_matrix, 1.0e-4
    )


def test__operated_mapping_matrix_property__with_operated_mapping_matrix_override(
    psf_3x3, rectangular_mapper_7x7_3x3
):
    psf = aa.m.MockPSF(operated_mapping_matrix=np.ones((2, 2)))

    operated_mapping_matrix_override = np.array([[1.0, 2.0], [3.0, 4.0]])

    linear_obj = aa.m.MockLinearObjFuncList(
        mapping_matrix=None,
        operated_mapping_matrix_override=operated_mapping_matrix_override,
    )

    inversion = aa.m.MockInversionImaging(
        mask=rectangular_mapper_7x7_3x3.mask,
        psf=psf,
        linear_obj_list=[rectangular_mapper_7x7_3x3, linear_obj],
    )

    operated_mapping_matrix_0 = np.array([[1.0, 1.0], [1.0, 1.0]])
    operated_mapping_matrix = np.array([[1.0, 1.0, 1.0, 2.0], [1.0, 1.0, 3.0, 4.0]])

    assert inversion.operated_mapping_matrix_list[0] == pytest.approx(
        operated_mapping_matrix_0, 1.0e-4
    )
    assert inversion.operated_mapping_matrix_list[1] == pytest.approx(
        operated_mapping_matrix_override, 1.0e-4
    )
    assert inversion.operated_mapping_matrix == pytest.approx(
        operated_mapping_matrix, 1.0e-4
    )


def test__curvature_matrix(rectangular_mapper_7x7_3x3):
    noise_map = np.ones(2)
    psf = aa.m.MockPSF(operated_mapping_matrix=np.ones((2, 10)))

    operated_mapping_matrix_override = np.array([[1.0, 2.0], [3.0, 4.0]])

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=1,
        mapping_matrix=None,
        operated_mapping_matrix_override=operated_mapping_matrix_override,
        regularization=None,
    )

    dataset = aa.DatasetInterface(
        data=aa.Array2D.ones(shape_native=(2, 10), pixel_scales=1.0),
        noise_map=noise_map,
        psf=psf,
    )

    inversion = aa.InversionImagingMapping(
        dataset=dataset,
        linear_obj_list=[linear_obj, rectangular_mapper_7x7_3x3],
        settings=aa.Settings(no_regularization_add_to_curvature_diag_value=False),
    )

    assert inversion.curvature_matrix[0:2, 0:2] == pytest.approx(
        np.array([[10.0, 14.0], [14.0, 20.0]]), 1.0e-4
    )

    assert inversion.curvature_matrix[0, 0] - 10.0 < 1.0e-12
    assert inversion.curvature_matrix[3, 3] - 2.0 < 1.0e-12

    inversion = aa.InversionImagingMapping(
        dataset=dataset,
        linear_obj_list=[linear_obj, rectangular_mapper_7x7_3x3],
        settings=aa.Settings(no_regularization_add_to_curvature_diag_value=True),
    )

    assert inversion.curvature_matrix[0, 0] - 10.0 > 0.0
    assert inversion.curvature_matrix[3, 3] - 2.0 < 1.0e-12


def test__mapping_matrix_over_sampled__delta_kernel_bins_to_mapping_matrix(
    rectangular_mapper_7x7_3x3,
):
    # Binning the sub-resolution mapping matrix rows by the mean of each sub-block
    # must reproduce the regular (sub_fraction folded) mapping matrix exactly.
    mapper = rectangular_mapper_7x7_3x3
    s = 2

    mapping_matrix = np.array(mapper.mapping_matrix)
    mapping_matrix_sub = np.array(mapper.mapping_matrix_over_sampled)

    assert mapping_matrix_sub.shape == (
        mapper.mask.pixels_in_mask * s**2,
        mapping_matrix.shape[1],
    )

    binned = mapping_matrix_sub.reshape(
        mapper.mask.pixels_in_mask, s**2, mapping_matrix.shape[1]
    ).mean(axis=1)

    assert binned == pytest.approx(mapping_matrix, abs=1.0e-14)


def _oversampled_psf_for(mask, s, kernel_n=5, sigma_frac=0.4):
    ps = mask.pixel_scales[0]
    c = (np.arange(kernel_n) - (kernel_n - 1) / 2.0) * (ps / s)
    yy, xx = np.meshgrid(-c, c, indexing="ij")
    kernel = np.exp(-0.5 * (yy**2 + xx**2) / (sigma_frac * ps) ** 2)
    kernel = kernel / kernel.sum()
    kernel = aa.Array2D.no_mask(values=kernel, pixel_scales=ps / s)
    return aa.Convolver(kernel=kernel, convolve_over_sample_size=s)


def test__operated_mapping_matrix__oversampled_psf__matches_brute_force(
    rectangular_mapper_7x7_3x3,
):
    # End-to-end mapping-formalism check: the oversampled operated mapping matrix
    # equals an independent brute-force fine-raster convolution + mean bin-down of
    # the sub-resolution mapping matrix (implemented with plain loops, not the
    # Convolver's own indexing).
    from scipy.signal import convolve as scipy_convolve

    mapper = rectangular_mapper_7x7_3x3
    mask = mapper.mask
    s = 2

    psf = _oversampled_psf_for(mask=mask, s=s)

    inversion = aa.m.MockInversionImaging(
        mask=mask, psf=psf, linear_obj_list=[mapper]
    )

    operated = np.array(inversion.operated_mapping_matrix_list[0])

    mapping_matrix_sub = np.array(mapper.mapping_matrix_over_sampled)
    n_src = mapping_matrix_sub.shape[1]

    mask_arr = np.array(mask)
    ny, nx = mask_arr.shape
    ys, xs = np.where(~mask_arr)

    native = np.zeros((ny * s, nx * s, n_src))
    k = 0
    for yi, xi in zip(ys, xs):
        for iy in range(s):
            for ix in range(s):
                native[yi * s + iy, xi * s + ix, :] = mapping_matrix_sub[k]
                k += 1

    kernel_fine = np.array(psf.kernel.native)
    convolved = np.zeros_like(native)
    for j in range(n_src):
        convolved[:, :, j] = scipy_convolve(
            native[:, :, j], kernel_fine, mode="same"
        )

    brute = convolved.reshape(ny, s, nx, s, n_src).mean(axis=(1, 3))[~mask_arr]

    assert operated == pytest.approx(brute, abs=1.0e-12)


def test__oversampled_psf__linear_func_and_preload_guards(
    rectangular_mapper_7x7_3x3,
):
    mapper = rectangular_mapper_7x7_3x3
    mask = mapper.mask

    psf = _oversampled_psf_for(mask=mask, s=2)

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=1, grid=None, mapping_matrix=np.ones((9, 1))
    )

    inversion = aa.m.MockInversionImaging(
        mask=mask, psf=psf, linear_obj_list=[linear_obj]
    )

    # Linear function objects are image-resolution; the oversampled path raises.
    with pytest.raises(exc.InversionException):
        inversion.operated_mapping_matrix_list

    inversion = aa.m.MockInversionImaging(
        mask=mask, psf=psf, linear_obj_list=[mapper]
    )

    # The kernel-native preload fast path raises too.
    with pytest.raises(exc.InversionException):
        inversion.data_linear_func_matrix_dict


def test__mapping_matrix_over_sampled_for__kxs__full_bin_reproduces_mapping_matrix(
    rectangular_mapper_7x7_3x3,
):
    # By linearity, the k x s matrix mean-binned s^2 -> 1 must equal the regular
    # (sub_fraction folded) mapping matrix exactly: the fold of the folds is the
    # full fold. The fixture's over sampler is uniform size 2, so s=1 puts k=2
    # (adaptive-free k x s) and s=2 puts k=1 (the 2b identity).
    mapper = rectangular_mapper_7x7_3x3
    mapping_matrix = np.array(mapper.mapping_matrix)
    n_pix = mapper.mask.pixels_in_mask

    # k=1 (s equals the evaluation size): identical to the 2b property.
    m_s2 = np.array(mapper.mapping_matrix_over_sampled_for(convolve_over_sample_size=2))
    assert m_s2 == pytest.approx(np.array(mapper.mapping_matrix_over_sampled), abs=0)

    # k=2 (s=1): one row per image pixel — must equal mapping_matrix exactly.
    m_s1 = np.array(mapper.mapping_matrix_over_sampled_for(convolve_over_sample_size=1))
    assert m_s1.shape == mapping_matrix.shape
    assert m_s1 == pytest.approx(mapping_matrix, abs=1.0e-14)

    # s=2 rows mean-binned to image resolution also reproduce mapping_matrix.
    binned = m_s2.reshape(n_pix, 4, mapping_matrix.shape[1]).mean(axis=1)
    assert binned == pytest.approx(mapping_matrix, abs=1.0e-14)


def test__operated_mapping_matrix_list__psf_convolution_performed_once_per_linear_obj(
    masked_imaging_7x7, delaunay_mapper_9_3x3, monkeypatch
):
    # The dense (mapping) route reaches `operated_mapping_matrix_list` from the
    # curvature matrix / data vector (via `operated_mapping_matrix`) and again from
    # `mapped_reconstructed_data_dict`. It must be cached so the PSF convolution of
    # each linear object's mapping matrix happens once per inversion.
    calls = []

    convolved_mapping_matrix_from = aa.Convolver.convolved_mapping_matrix_from

    def counted(self, *args, **kwargs):
        calls.append(1)
        return convolved_mapping_matrix_from(self, *args, **kwargs)

    monkeypatch.setattr(aa.Convolver, "convolved_mapping_matrix_from", counted)

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[delaunay_mapper_9_3x3],
    )

    assert isinstance(inversion, aa.InversionImagingMapping)

    inversion.log_det_curvature_reg_matrix_term
    inversion.reconstruction
    inversion.mapped_reconstructed_operated_data
    inversion.mapped_reconstructed_data

    assert len(calls) == 1


requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="the imaging sparse operator is built with jax",
)


def _mapper_with_invalid_slots_from(mapper):
    """
    Returns a copy of `mapper` whose stencil table carries every kind of entry
    `mapping_matrix_from` drops, all with non-zero weights:

    - an unused slot (slot >= pix_sizes) padded with index -1 (odd sub-pixels of a Delaunay row);
    - a negative index inside a used slot (even sub-pixels, whose pix_size is bumped over a -1 slot);
    - an extra unused slot holding a real source pixel index (every sub-pixel).

    The dense mapping matrix is unchanged by these entries, so a correct sparse path must be too.
    """
    pix_indexes = np.asarray(mapper.pix_indexes_for_sub_slim_index)
    pix_weights = np.asarray(mapper.pix_weights_for_sub_slim_index, dtype=float)
    pix_sizes = np.asarray(mapper.pix_sizes_for_sub_slim_index).copy()

    M_sub, P = pix_indexes.shape

    unused = np.arange(P)[None, :] >= pix_sizes[:, None]

    assert unused.any()
    assert (pix_indexes[unused] == -1).all()

    pix_weights = np.where(unused, 0.7, pix_weights)

    bump = (np.arange(M_sub) % 2 == 0) & (pix_sizes < P)
    pix_sizes[bump] += 1

    pix_indexes = np.concatenate(
        [pix_indexes, (np.arange(M_sub) % mapper.params)[:, None]], axis=1
    )
    pix_weights = np.concatenate([pix_weights, np.full((M_sub, 1), 0.9)], axis=1)

    class MapperWithInvalidSlots(type(mapper)):
        @property
        def pix_indexes_for_sub_slim_index(self):
            return pix_indexes

        @property
        def pix_weights_for_sub_slim_index(self):
            return pix_weights

        @property
        def pix_sizes_for_sub_slim_index(self):
            return pix_sizes

    corrupted = MapperWithInvalidSlots(
        interpolator=mapper.interpolator,
        regularization=mapper.regularization,
        image_plane_mesh_grid=mapper.image_plane_mesh_grid,
    )

    assert (corrupted.pix_indexes_for_sub_slim_index < 0).any()
    assert np.allclose(corrupted.mapping_matrix, mapper.mapping_matrix)

    return corrupted


@requires_jax
def test__sparse__mapper_with_invalid_stencil_slots__matches_mapping_inversion(
    masked_imaging_7x7, delaunay_mapper_9_3x3
):
    mapper = _mapper_with_invalid_slots_from(delaunay_mapper_9_3x3)

    settings = aa.Settings(use_positive_only_solver=False)

    inversion_sparse = aa.Inversion(
        dataset=masked_imaging_7x7.apply_sparse_operator(),
        linear_obj_list=[mapper],
        settings=settings,
    )
    inversion_mapping = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[mapper],
        settings=settings,
    )

    assert isinstance(inversion_sparse, InversionImagingSparse)

    assert np.asarray(inversion_sparse.data_vector) == pytest.approx(
        np.asarray(inversion_mapping.data_vector), 1.0e-6
    )
    assert np.asarray(inversion_sparse.curvature_matrix) == pytest.approx(
        np.asarray(inversion_mapping.curvature_matrix), 1.0e-6
    )

    reconstruction = np.asarray(inversion_sparse.reconstruction)

    assert reconstruction == pytest.approx(
        np.asarray(inversion_mapping.reconstruction), 1.0e-4
    )

    mapped = np.asarray(inversion_sparse.mapped_reconstructed_data_dict[mapper].array)

    assert mapped == pytest.approx(
        np.asarray(mapper.mapping_matrix) @ reconstruction, 1.0e-8
    )
    assert mapped == pytest.approx(
        np.asarray(inversion_mapping.mapped_reconstructed_data_dict[mapper].array),
        1.0e-4,
    )
    assert np.asarray(
        inversion_sparse.mapped_reconstructed_operated_data_dict[mapper].array
    ) == pytest.approx(
        np.asarray(
            inversion_mapping.mapped_reconstructed_operated_data_dict[mapper].array
        ),
        1.0e-4,
    )


@requires_jax
def test__sparse__mapper_with_invalid_stencil_slots__matches_mapping_inversion__jax(
    masked_imaging_7x7, delaunay_mapper_9_3x3
):
    import jax

    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp

    mapper = _mapper_with_invalid_slots_from(delaunay_mapper_9_3x3)
    mapper_jax = type(mapper)(
        interpolator=mapper.interpolator,
        regularization=mapper.regularization,
        image_plane_mesh_grid=mapper.image_plane_mesh_grid,
        xp=jnp,
    )

    settings = aa.Settings(use_positive_only_solver=False)

    inversion_sparse = aa.Inversion(
        dataset=masked_imaging_7x7.apply_sparse_operator(),
        linear_obj_list=[mapper_jax],
        settings=settings,
        xp=jnp,
    )
    inversion_mapping = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[mapper],
        settings=settings,
    )

    assert isinstance(inversion_sparse, InversionImagingSparse)

    assert np.asarray(inversion_sparse.data_vector) == pytest.approx(
        np.asarray(inversion_mapping.data_vector), 1.0e-6
    )
    assert np.asarray(inversion_sparse.curvature_matrix) == pytest.approx(
        np.asarray(inversion_mapping.curvature_matrix), 1.0e-6
    )

    reconstruction = np.asarray(inversion_sparse.reconstruction)

    assert np.asarray(
        inversion_sparse.mapped_reconstructed_data_dict[mapper_jax].array
    ) == pytest.approx(np.asarray(mapper.mapping_matrix) @ reconstruction, 1.0e-8)


def test__sparse_helpers__import_and_run_with_jax_blocked():
    """
    The mapper triplets and the two imaging scatter-add helpers (`data_vector_via_psf_weighted_data_from`,
    `mapped_reconstructed_operated_data_via_sparse_operator_from`) must work in the `unittest-nojax`
    leg. JAX is in `sys.modules` for the rest of a normal test session, so a subprocess hides `jax`,
    `jaxlib` and `nufftax` from the import system and checks them against the dense mapping matrix
    on a Delaunay mapper, whose unused stencil slots carry `-1` indices.

    The `ImagingSparseOperator` itself (FFT precompute and curvature blocks) is JAX-only, so the full
    `InversionImagingSparse` is not run here.
    """
    script = textwrap.dedent(
        """
        import importlib.machinery
        import sys

        _find_spec = importlib.machinery.PathFinder.find_spec.__func__


        def _find_spec_blocking_jax(cls, name, path=None, target=None):
            if name.split(".")[0] in ("jax", "jaxlib", "nufftax"):
                return None
            return _find_spec(cls, name, path, target)


        importlib.machinery.PathFinder.find_spec = classmethod(_find_spec_blocking_jax)

        # Guard the guard: if the block stopped working this test would pass vacuously.
        try:
            import jax
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("jax was not blocked")

        import numpy as np

        import autoarray as aa
        from autoarray import fixtures
        from autoarray.inversion.inversion.imaging import inversion_imaging_util

        mapper = fixtures.make_delaunay_mapper_9_3x3()
        mask = fixtures.make_mask_2d_7x7()

        assert (np.asarray(mapper.pix_indexes_for_sub_slim_index) < 0).any()

        mapping_matrix = np.asarray(mapper.mapping_matrix)

        rows, cols, vals = mapper.sparse_triplets_data

        rng = np.random.default_rng(seed=1)
        psf_weighted_data = rng.normal(size=mapping_matrix.shape[0])

        data_vector = inversion_imaging_util.data_vector_via_psf_weighted_data_from(
            psf_weighted_data=psf_weighted_data,
            rows=rows,
            cols=cols,
            vals=vals,
            S=mapper.params,
            xp=np,
        )

        assert np.allclose(data_vector, mapping_matrix.T @ psf_weighted_data)

        rows, cols, vals = mapper.sparse_triplets_curvature

        reconstruction = rng.normal(size=mapper.params)

        mapped = inversion_imaging_util.mapped_reconstructed_operated_data_via_sparse_operator_from(
            reconstruction=reconstruction,
            rows=rows,
            cols=cols,
            vals=vals,
            fft_index_for_masked_pixel=mask.fft_index_for_masked_pixel,
            data_shape=mask.shape_native,
            xp=np,
        )

        assert np.allclose(mapped, mapping_matrix @ reconstruction)

        assert not any(
            name.split(".")[0] in ("jax", "jaxlib", "nufftax") for name in sys.modules
        )
        """
    )

    environment = dict(os.environ)
    environment["PYAUTO_DISABLE_JAX"] = "1"

    subprocess.run(
        [sys.executable, "-c", script], check=True, env=environment, timeout=600
    )

# ----------------------------------------------------------------------------
# Sparse transposed-blur curvature route (`Settings.use_sparse_curvature`):
# `InversionImagingMapping` must return the same `F` as the dense product.
# ----------------------------------------------------------------------------

from autoarray.inversion.inversion.imaging.mapping import InversionImagingMapping

requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="requires jax (installed via the [optional] extras; absent on the NumPy-only matrix env)",
)


def _func_list_from(mask):
    grid = aa.Grid2D.from_mask(mask=mask)
    mapping_matrix = np.full(fill_value=0.5, shape=(grid.shape[0], 2))
    mapping_matrix[0, 0] = 0.8
    mapping_matrix[1, 1] = 0.4
    return aa.m.MockLinearObjFuncList(
        parameters=2, grid=grid, mapping_matrix=mapping_matrix
    )


def _off_centre_imaging_and_mapper(kernel_shape):
    """
    A mask whose bounding box sits off the stamp centre, under an even- or odd-shaped
    asymmetric kernel, so the roll / slice conventions of the transposed FFT blur are
    exercised rather than cancelling by symmetry. The mapper is a 3x3 rectangular mesh
    over the sub-size-2 grid of that mask.
    """
    from autoarray.inversion.mesh.mesh.rectangular_rtu_adapt_density import (
        overlay_grid_from,
    )

    mask = np.full((9, 11), True)
    mask[1:5, 2:7] = False
    mask[4, 6] = True
    mask = aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))

    rng = np.random.default_rng(3)
    kernel = aa.Array2D.no_mask(
        values=rng.uniform(0.1, 1.0, size=kernel_shape), pixel_scales=(1.0, 1.0)
    )
    # Even kernels are FFT-only, and the FFT path is the production one anyway.
    psf = aa.Convolver(kernel=kernel, use_fft=True)

    imaging = aa.Imaging(
        data=aa.Array2D(values=rng.normal(size=mask.shape), mask=mask),
        noise_map=aa.Array2D(values=rng.uniform(1.0, 3.0, size=mask.shape), mask=mask),
        psf=psf,
        over_sample_size_lp=1,
    ).apply_mask(mask=mask)

    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=2)
    shape_native = (3, 3)
    source_plane_mesh_grid = overlay_grid_from(
        shape_native=shape_native, grid=grid.over_sampled
    )
    mesh = aa.mesh.RectangularUniform(shape=shape_native)
    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=aa.Grid2DIrregular(source_plane_mesh_grid),
        adapt_data=aa.Array2D.ones(shape_native, pixel_scales=0.1),
    )
    mapper = aa.Mapper(
        interpolator=interpolator, regularization=aa.reg.Constant(coefficient=1.0)
    )
    return imaging, mapper


def _assert_sparse_matches_dense(dense, sparse, rel=1.0e-10):
    assert isinstance(sparse, InversionImagingMapping)
    assert sparse._use_sparse_curvature
    assert not dense._use_sparse_curvature

    F_dense = np.asarray(dense.curvature_matrix)
    F_sparse = np.asarray(sparse.curvature_matrix)

    assert F_sparse.dtype == np.float64
    assert F_sparse == pytest.approx(F_dense, rel=rel, abs=rel * np.abs(F_dense).max())
    assert np.asarray(sparse.curvature_reg_matrix) == pytest.approx(
        np.asarray(dense.curvature_reg_matrix), rel=rel, abs=rel * np.abs(F_dense).max()
    )
    assert np.asarray(sparse.reconstruction) == pytest.approx(
        np.asarray(dense.reconstruction), rel=1.0e-6, abs=1.0e-8
    )
    assert float(sparse.log_det_curvature_reg_matrix_term) == pytest.approx(
        float(dense.log_det_curvature_reg_matrix_term), rel=1.0e-8
    )


@pytest.mark.parametrize("positive_only", [False, True])
def test__sparse_curvature__numpy__mapper_and_func_list_match_dense(
    masked_imaging_7x7, rectangular_mapper_7x7_3x3, positive_only
):
    func = _func_list_from(masked_imaging_7x7.mask)

    for linear_obj_list in (
        [rectangular_mapper_7x7_3x3],
        [func, rectangular_mapper_7x7_3x3],
        [rectangular_mapper_7x7_3x3, func],
    ):
        dense = aa.Inversion(
            dataset=masked_imaging_7x7,
            linear_obj_list=linear_obj_list,
            settings=aa.Settings(use_positive_only_solver=positive_only),
        )
        # On NumPy the route is taken only when asked for explicitly (it is the reference
        # implementation there, not a speed-up).
        sparse = aa.Inversion(
            dataset=masked_imaging_7x7,
            linear_obj_list=linear_obj_list,
            settings=aa.Settings(
                use_positive_only_solver=positive_only, use_sparse_curvature=True
            ),
        )
        _assert_sparse_matches_dense(dense=dense, sparse=sparse)


def test__sparse_curvature__column_blocked_transposed_blur_matches_dense(
    masked_imaging_7x7, rectangular_mapper_7x7_3x3, monkeypatch
):
    """
    The transposed blur runs in column blocks; with a block smaller than the parameter count
    (9 mesh + 2 func columns, blocks of 4: two full blocks and a partial one) the result is
    unchanged.
    """
    from autoarray.inversion.inversion.imaging import inversion_imaging_util

    monkeypatch.setattr(inversion_imaging_util, "SPARSE_TRANSPOSE_BLUR_COLUMNS", 4)

    func = _func_list_from(masked_imaging_7x7.mask)
    dense = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[func, rectangular_mapper_7x7_3x3],
        settings=aa.Settings(use_positive_only_solver=False),
    )
    sparse = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[func, rectangular_mapper_7x7_3x3],
        settings=aa.Settings(use_positive_only_solver=False, use_sparse_curvature=True),
    )
    _assert_sparse_matches_dense(dense=dense, sparse=sparse)


def test__sparse_curvature__no_mapper_falls_back_to_dense(masked_imaging_7x7):
    func = _func_list_from(masked_imaging_7x7.mask)

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[func],
        settings=aa.Settings(use_sparse_curvature=True),
    )

    assert not inversion._use_sparse_curvature
    assert np.asarray(inversion.curvature_matrix).shape == (2, 2)


@pytest.mark.parametrize("kernel_shape", [(3, 3), (4, 4), (3, 5), (5, 4)])
def test__sparse_curvature__numpy__off_centre_mask_even_and_odd_kernels(kernel_shape):
    imaging, mapper = _off_centre_imaging_and_mapper(kernel_shape=kernel_shape)
    func = _func_list_from(imaging.mask)

    dense = aa.Inversion(
        dataset=imaging,
        linear_obj_list=[func, mapper],
        settings=aa.Settings(use_positive_only_solver=False),
    )
    sparse = aa.Inversion(
        dataset=imaging,
        linear_obj_list=[func, mapper],
        settings=aa.Settings(use_positive_only_solver=False, use_sparse_curvature=True),
    )
    _assert_sparse_matches_dense(dense=dense, sparse=sparse)


@requires_jax
@pytest.mark.parametrize("kernel_shape", [(3, 3), (4, 4), (3, 5), (5, 4)])
def test__sparse_curvature__jax_fp64__matches_dense_exactly(kernel_shape):
    """
    JAX FFT blurs in fp64 end to end (`use_mixed_precision=False`): the sparse route is the
    dense product to round-off, under even and odd kernels on an off-centre mask.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    imaging, mapper = _off_centre_imaging_and_mapper(kernel_shape=kernel_shape)
    func = _func_list_from(imaging.mask)
    mapper_jax = type(mapper)(
        interpolator=mapper.interpolator,
        regularization=mapper.regularization,
        image_plane_mesh_grid=mapper.image_plane_mesh_grid,
        xp=jnp,
    )

    dense = aa.Inversion(
        dataset=imaging,
        linear_obj_list=[func, mapper_jax],
        settings=aa.Settings(use_positive_only_solver=False),
        xp=jnp,
    )
    sparse = aa.Inversion(
        dataset=imaging,
        linear_obj_list=[func, mapper_jax],
        settings=aa.Settings(use_positive_only_solver=False, use_sparse_curvature=True),
        xp=jnp,
    )
    _assert_sparse_matches_dense(dense=dense, sparse=sparse)

    # The fp64 NumPy dense product is the same matrix again.
    dense_np = aa.Inversion(
        dataset=imaging,
        linear_obj_list=[func, mapper],
        settings=aa.Settings(use_positive_only_solver=False),
    )
    F_np = np.asarray(dense_np.curvature_matrix)
    assert np.asarray(sparse.curvature_matrix) == pytest.approx(
        F_np, rel=1.0e-10, abs=1.0e-10 * np.abs(F_np).max()
    )


@requires_jax
@pytest.mark.parametrize("kernel_shape", [(3, 3), (4, 4)])
@pytest.mark.parametrize("transposed_mixed", [False, True])
def test__sparse_curvature__jax_mixed_precision__within_fp32_of_dense(
    kernel_shape, transposed_mixed
):
    """
    The production configuration: `use_mixed_precision=True` gives the forward blur an fp32
    cube and complex64 forward FFT, so `B` itself carries ~1e-7 relative error and the two
    routes can only agree to that level (the dense product squares `B`, the sparse route uses
    it once). With `sparse_curvature_mixed_precision` the transposed blur takes the same fp32
    path; the test pins that it runs and stays at the fp32 level, not that it is exact.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    imaging, mapper = _off_centre_imaging_and_mapper(kernel_shape=kernel_shape)
    func = _func_list_from(imaging.mask)
    mapper_jax = type(mapper)(
        interpolator=mapper.interpolator,
        regularization=mapper.regularization,
        image_plane_mesh_grid=mapper.image_plane_mesh_grid,
        xp=jnp,
    )

    dense = aa.Inversion(
        dataset=imaging,
        linear_obj_list=[func, mapper_jax],
        settings=aa.Settings(use_positive_only_solver=False, use_mixed_precision=True),
        xp=jnp,
    )
    sparse = aa.Inversion(
        dataset=imaging,
        linear_obj_list=[func, mapper_jax],
        settings=aa.Settings(
            use_positive_only_solver=False,
            use_mixed_precision=True,
            use_sparse_curvature=True,
            sparse_curvature_mixed_precision=transposed_mixed,
        ),
        xp=jnp,
    )

    assert sparse._use_sparse_curvature

    F_dense = np.asarray(dense.curvature_matrix)
    F_sparse = np.asarray(sparse.curvature_matrix)

    assert F_sparse.dtype == np.float64
    assert F_sparse == pytest.approx(
        F_dense, rel=1.0e-5, abs=1.0e-5 * np.abs(F_dense).max()
    )
    # The toy system (19 pixels, a 3x3 mesh, random data, the positive-negative solve) is
    # ill-conditioned enough to amplify the 1e-7 of `F` by three orders of magnitude in the
    # reconstruction; this is the fp32 amplification the fp64 default guards against.
    assert np.asarray(sparse.reconstruction) == pytest.approx(
        np.asarray(dense.reconstruction), rel=1.0e-2, abs=1.0e-6
    )


@requires_jax
def test__sparse_curvature__jax__config_default_off_and_numpy_ignores_config(
    masked_imaging_7x7, rectangular_mapper_7x7_3x3
):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    assert aa.Settings().use_sparse_curvature is False
    assert aa.Settings().sparse_curvature_mixed_precision is False

    mapper_jax = type(rectangular_mapper_7x7_3x3)(
        interpolator=rectangular_mapper_7x7_3x3.interpolator,
        regularization=rectangular_mapper_7x7_3x3.regularization,
        image_plane_mesh_grid=rectangular_mapper_7x7_3x3.image_plane_mesh_grid,
        xp=jnp,
    )

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7, linear_obj_list=[mapper_jax], xp=jnp
    )
    assert not inversion._use_sparse_curvature

    # A config-level `True` reaches a JAX inversion but not a NumPy one.
    class _CfgOn(aa.Settings):
        @property
        def use_sparse_curvature(self):
            return True  # as a `general.yaml` `use_sparse_curvature: true` would read

    jax_on = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[mapper_jax],
        settings=_CfgOn(),
        xp=jnp,
    )
    numpy_cfg = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[rectangular_mapper_7x7_3x3],
        settings=_CfgOn(),
    )
    assert jax_on._use_sparse_curvature
    assert not numpy_cfg._use_sparse_curvature
