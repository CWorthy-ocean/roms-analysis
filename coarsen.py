from dataclasses import dataclass
import xarray as xr

@dataclass(frozen=True, kw_only=True)
class ROMSCoarsener:
    """
    Coarsens fields on the ROMS C-grid.

    Parameters
    ----------
    ds_grid : xr.Dataset
        Contains the grid information and precomputed metrics.
    """

    ds_grid: xr.Dataset

    def __post_init__(self):
        """Compute all necessary grid metrics for length- and area-weighted coarsening."""
        self.ds_grid["dxi_rho"] = 1 / self.ds_grid["pm"]
        self.ds_grid["deta_rho"] = 1 / self.ds_grid["pn"]
        self.ds_grid["dxi_u"] = (0.5 * (self.ds_grid["dxi_rho"] + self.ds_grid["dxi_rho"].shift(xi_rho=1))
                                 .isel(xi_rho=slice(1, None))
                                 .swap_dims({"xi_rho": "xi_u"}))
        self.ds_grid["deta_u"] = (0.5 * (self.ds_grid["deta_rho"] + self.ds_grid["deta_rho"].shift(xi_rho=1))
                                  .isel(xi_rho=slice(1, None))
                                  .swap_dims({"xi_rho": "xi_u"}))
        self.ds_grid["dxi_v"] = (0.5 * (self.ds_grid["dxi_rho"] + self.ds_grid["dxi_rho"].shift(eta_rho=1))
                                 .isel(eta_rho=slice(1, None))
                                 .swap_dims({"eta_rho": "eta_v"}))
        self.ds_grid["deta_v"] = (0.5 * (self.ds_grid["deta_rho"] + self.ds_grid["deta_rho"].shift(eta_rho=1))
                                  .isel(eta_rho=slice(1, None))
                                  .swap_dims({"eta_rho": "eta_v"}))
        self.ds_grid["area_rho"] = self.ds_grid["dxi_rho"] * self.ds_grid["deta_rho"]
        self.ds_grid["area_u"] = self.ds_grid["dxi_u"] * self.ds_grid["deta_u"]
        self.ds_grid["area_v"] = self.ds_grid["dxi_v"] * self.ds_grid["deta_v"]

    def coarsen_t(self, var, factor):
        """
        Coarsen a field on the rho grid.
        - Inner grid points are coarsened via area-weighted mean of factor x factor grid points.
        - Boundary grid points are coarsened via length-weighted mean of factor x 1 or 1 x factor grid points.
        - Corner grid points are set equal to the corner grid points of the original grid.

        Parameters
        ----------
        var : xr.DataArray
            The variable to be coarsened.
        factor : int
            The coarsening factor.

        Returns
        -------
        xr.DataArray
            The coarsened variable.
        """
        if (len(var.eta_rho) - 2) % factor != 0:
            raise ValueError(f"Dimension (len(eta_rho) - 2) must be divisible by factor {factor}.")
        if (len(var.xi_rho) - 2) % factor != 0:
            raise ValueError(f"Dimension (len(xi_rho) - 2) must be divisible by factor {factor}.")

        var_inner = var.isel(eta_rho=slice(1, -1), xi_rho=slice(1, -1))
        area = self.ds_grid["area_rho"].isel(eta_rho=slice(1, -1), xi_rho=slice(1, -1))                                               
        var_inner_coarse = (
            (var_inner * area).coarsen(eta_rho=factor, xi_rho=factor).sum()
        ) / area.coarsen(eta_rho=factor, xi_rho=factor).sum()

        var_west = var.isel(eta_rho=slice(1, -1), xi_rho=0)
        var_east = var.isel(eta_rho=slice(1, -1), xi_rho=-1)
        var_south = var.isel(eta_rho=0, xi_rho=slice(1, -1))
        var_north = var.isel(eta_rho=-1, xi_rho=slice(1, -1))
        length_west = self.ds_grid["deta_rho"].isel(eta_rho=slice(1, -1), xi_rho=0)
        length_east = self.ds_grid["deta_rho"].isel(eta_rho=slice(1, -1), xi_rho=-1)
        length_south = self.ds_grid["dxi_rho"].isel(eta_rho=0, xi_rho=slice(1, -1))
        length_north = self.ds_grid["dxi_rho"].isel(eta_rho=-1, xi_rho=slice(1, -1))

        var_west_coarse = (
            (var_west * length_west).coarsen(eta_rho=factor).sum()
        ) / length_west.coarsen(eta_rho=factor).sum()
        var_east_coarse = (
            (var_east * length_east).coarsen(eta_rho=factor).sum()
        ) / length_east.coarsen(eta_rho=factor).sum()
        var_south_coarse = (
            (var_south * length_south).coarsen(xi_rho=factor).sum()
        ) / length_south.coarsen(xi_rho=factor).sum()
        var_north_coarse = (
            (var_north * length_north).coarsen(xi_rho=factor).sum()
        ) / length_north.coarsen(xi_rho=factor).sum()   

        var_coarse = xr.concat([var_south_coarse, var_inner_coarse, var_north_coarse], dim="eta_rho")
        var_west_coarse = xr.concat([var.isel(eta_rho=0, xi_rho=0), var_west_coarse, var.isel(eta_rho=-1, xi_rho=0)], dim="eta_rho")
        var_east_coarse = xr.concat([var.isel(eta_rho=0, xi_rho=-1), var_east_coarse, var.isel(eta_rho=-1, xi_rho=-1)], dim="eta_rho")
        var_coarse = xr.concat([var_west_coarse, var_coarse, var_east_coarse], dim="xi_rho")
        
        return var_coarse

    def coarsen_u(self, var, factor):
        """
        Coarsen a field on the u-grid.
        - Inner grid points are coarsened via length-weighted mean of 1 x factor grid points, consistent with C-grid velocity interpretation.
        - Southernmost and northernmost grid points are set equal to original grid points (at same locations).


        Parameters
        ----------
        var : xr.DataArray
            The variable to be coarsened.
        factor : int
            The coarsening factor.

        Returns
        -------
        xr.DataArray
            The coarsened variable.
        """
        if (len(var.eta_rho) - 2) % factor != 0:
            raise ValueError(f"Dimension (len(eta_rho) - 2) must be divisible by factor {factor}.")
        if (len(var.xi_u) - 1) % factor != 0:
            raise ValueError(f"Dimension (len(xi_u) - 1) must be divisible by factor {factor}.")

        var_downsampled = var.isel(xi_u=slice(0, None, factor))
        deta_u_downsampled = self.ds_grid["deta_u"].isel(xi_u=slice(0, None, factor))
        var_inner = var_downsampled.isel(eta_rho=slice(1, -1))
        deta_u_inner = deta_u_downsampled.isel(eta_rho=slice(1, -1))
        
        var_inner_coarse = (
            (var_inner * deta_u_inner).coarsen(eta_rho=factor).sum()
        ) / deta_u_inner.coarsen(eta_rho=factor).sum()

        var_coarse = xr.concat([var_downsampled.isel(eta_rho=0), var_inner_coarse, var_downsampled.isel(eta_rho=-1)], dim="eta_rho")
        var_coarse = var_coarse.transpose()
        return var_coarse

    def coarsen_v(self, var, factor):
        """
        Coarsen a field on the v-grid.
        - Inner grid points are coarsened via length-weighted mean of factor x 1 factor grid points, consistent with C-grid velocity interpretation.
        - Westernmost and easternmost grid points are set equal to original grid points (at same locations).

        Parameters
        ----------
        var : xr.DataArray
            The variable to be coarsened.
        factor : int
            The coarsening factor.

        Returns
        -------
        xr.DataArray
            The coarsened variable.
        """
        if (len(var.eta_v) - 1) % factor != 0:
            raise ValueError(f"Dimension (len(eta_v) - 1) must be divisible by factor {factor}.")
        if (len(var.xi_rho) - 2) % factor != 0:
            raise ValueError(f"Dimension (len(xi_rho) - 2) must be divisible by factor {factor}.")

        var_downsampled = var.isel(eta_v=slice(0, None, factor))
        dxi_v_downsampled = self.ds_grid["dxi_v"].isel(eta_v=slice(0, None, factor))
        var_inner = var_downsampled.isel(xi_rho=slice(1, -1))
        dxi_v_inner = dxi_v_downsampled.isel(xi_rho=slice(1, -1))
        
        var_inner_coarse = (
            (var_inner * dxi_v_inner).coarsen(xi_rho=factor).sum()
        ) / dxi_v_inner.coarsen(xi_rho=factor).sum()

        var_coarse = xr.concat([var_downsampled.isel(xi_rho=0), var_inner_coarse, var_downsampled.isel(xi_rho=-1)], dim="xi_rho")
        return var_coarse

    def __call__(self, var, factor):
        """
        Coarsen a field based on whether it is a tracer, u-, or v-field.

        Parameters
        ----------
        var : xr.DataArray
            The variable to be coarsened.
        factor : int
            The coarsening factor.

        Returns
        -------
        xr.DataArray
            The coarsened variable.

        Raises
        ------
        ValueError
            If the variable dimensions do not match any expected grid type.
        """
        dims = set(var.dims)
        if {'eta_rho', 'xi_rho'}.issubset(dims):
            return self.coarsen_t(var, factor)
        elif {'eta_rho', 'xi_u'}.issubset(dims):
            return self.coarsen_u(var, factor)
        elif {'eta_v', 'xi_rho'}.issubset(dims):
            return self.coarsen_v(var, factor)
        else:
            raise ValueError("Variable dimensions do not match any expected grid type.")
