from dataclasses import dataclass
import xarray as xr
import numpy as np
from typing import Optional

@dataclass(frozen=True, kw_only=True)
class ROMSCoarsener:
    """
    Coarsens fields on the ROMS C-grid.

    Parameters
    ----------
    ds_grid : xr.Dataset
        Contains the grid information and precomputed metrics.
    N : Optional[int]
        The number of vertical levels. Only necessary for thickness-weighted coarsening.
    theta_s : Optional[float]
        The surface control parameter. Must satisfy 0 < theta_s <= 10. Only necessary for thickness-weighted coarsening.
    theta_b : Optional[float]
        The bottom control parameter. Must satisfy 0 < theta_b <= 4. Only necessary for thickness-weighted coarsening.
    hc : Optional[float]
        The critical depth. Only necessary for thickness-weighted coarsening.
    """

    ds_grid: xr.Dataset
    N: Optional[int] = None
    theta_s: Optional[float] = None
    theta_b: Optional[float] = None
    hc: Optional[float] = None

    def __post_init__(self):
        """Compute all necessary grid metrics for length- and area-weighted coarsening."""

        ds = xr.Dataset()
        ds["dxi_rho"] = 1 / self.ds_grid["pm"]
        ds["deta_rho"] = 1 / self.ds_grid["pn"]

        ds["dxi_u"] = (0.5 * (ds["dxi_rho"] + ds["dxi_rho"].shift(xi_rho=1))
                 .isel(xi_rho=slice(1, None))
                 .swap_dims({"xi_rho": "xi_u"}))
        ds["deta_u"] = (0.5 * (ds["deta_rho"] + ds["deta_rho"].shift(xi_rho=1))
                  .isel(xi_rho=slice(1, None))
                  .swap_dims({"xi_rho": "xi_u"}))
        ds["dxi_v"] = (0.5 * (ds["dxi_rho"] + ds["dxi_rho"].shift(eta_rho=1))
                 .isel(eta_rho=slice(1, None))
                 .swap_dims({"eta_rho": "eta_v"}))
        ds["deta_v"] = (0.5 * (ds["deta_rho"] + ds["deta_rho"].shift(eta_rho=1))
                  .isel(eta_rho=slice(1, None))
                  .swap_dims({"eta_rho": "eta_v"}))

        ds["area_rho"] = ds["dxi_rho"] * ds["deta_rho"]

        # Conditional computation for thickness-weighted coarsening
        if all(v is not None for v in [self.N, self.theta_s, self.theta_b, self.hc]):
            cs_w, sigma_w = sigma_stretch(self.theta_s, self.theta_b, self.N, "w")
            zw = compute_depth(self.ds_grid["h"] * 0, self.ds_grid["h"], self.hc, cs_w, sigma_w)
            dz = zw.diff(dim="s_w").rename({"s_w": "s_rho"})

            ds["dz"] = dz
            ds["dzu"] = interpolate_from_rho_to_u(dz)
            ds["dzv"] = interpolate_from_rho_to_v(dz)
            ds = ds.drop_vars(['eta_rho', 'xi_rho'])
        
        # Store in an internal attribute (not modifying the original ds_grid)
        object.__setattr__(self, 'ds', ds)

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
        area = self.ds["area_rho"].isel(eta_rho=slice(1, -1), xi_rho=slice(1, -1))                                               
        var_inner_coarse = (
            (var_inner * area).coarsen(eta_rho=factor, xi_rho=factor).sum()
        ) / area.coarsen(eta_rho=factor, xi_rho=factor).sum()

        var_west = var.isel(eta_rho=slice(1, -1), xi_rho=0)
        var_east = var.isel(eta_rho=slice(1, -1), xi_rho=-1)
        var_south = var.isel(eta_rho=0, xi_rho=slice(1, -1))
        var_north = var.isel(eta_rho=-1, xi_rho=slice(1, -1))
        length_west = self.ds["deta_rho"].isel(eta_rho=slice(1, -1), xi_rho=0)
        length_east = self.ds["deta_rho"].isel(eta_rho=slice(1, -1), xi_rho=-1)
        length_south = self.ds["dxi_rho"].isel(eta_rho=0, xi_rho=slice(1, -1))
        length_north = self.ds["dxi_rho"].isel(eta_rho=-1, xi_rho=slice(1, -1))

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
        deta_u_downsampled = self.ds["deta_u"].isel(xi_u=slice(0, None, factor))
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
        dxi_v_downsampled = self.ds["dxi_v"].isel(eta_v=slice(0, None, factor))
        var_inner = var_downsampled.isel(xi_rho=slice(1, -1))
        dxi_v_inner = dxi_v_downsampled.isel(xi_rho=slice(1, -1))
        
        var_inner_coarse = (
            (var_inner * dxi_v_inner).coarsen(xi_rho=factor).sum()
        ) / dxi_v_inner.coarsen(xi_rho=factor).sum()

        var_coarse = xr.concat([var_downsampled.isel(xi_rho=0), var_inner_coarse, var_downsampled.isel(xi_rho=-1)], dim="xi_rho")
        return var_coarse

    def __call__(self, var, factor, thickness_weighted=False):
        """
        Coarsen a field based on whether it is a tracer, u-, or v-field.
    
        Parameters
        ----------
        var : xr.DataArray
            The variable to be coarsened.
        factor : int
            The coarsening factor.
        thickness_weighted : bool, optional
            If True, performs thickness-weighted coarsening. Defaults to False.
    
        Returns
        -------
        xr.DataArray
            The coarsened variable.
    
        Raises
        ------
        ValueError
            If the variable dimensions do not match any expected grid type.
        ValueError
            If thickness-weighted coarsening is requested but the necessary thickness variables are not present.
        """
    
        dims = set(var.dims)
    
        # Check for thickness-weighted coarsening
        if thickness_weighted:
            # Ensure necessary thickness variables are available
            if not all(v in self.ds for v in ["dz", "dzu", "dzv"]):
                raise ValueError(
                    "Thickness-weighted coarsening requires the presence of the thicknesses 'dz', 'dzu', and 'dzv'. "
                    "Ensure that ROMSCoarsener is initialized with the required parameters (N, theta_s, theta_b, hc)."
                )
    
        if {'eta_rho', 'xi_rho'}.issubset(dims):
            # Coarsen tracer field
            if thickness_weighted:
                dz = self.ds["dz"]
                return (self.coarsen_t(var * dz, factor) / self.coarsen_t(dz, factor))
            else:
                return self.coarsen_t(var, factor)
    
        elif {'eta_rho', 'xi_u'}.issubset(dims):
            # Coarsen u-field
            if thickness_weighted:
                dzu = self.ds["dzu"]
                return (self.coarsen_u(var * dzu, factor) / self.coarsen_u(dzu, factor))
            else:
                return self.coarsen_u(var, factor)
    
        elif {'eta_v', 'xi_rho'}.issubset(dims):
            # Coarsen v-field
            if thickness_weighted:
                dzv = self.ds["dzv"]
                return (self.coarsen_v(var * dzv, factor) / self.coarsen_v(dzv, factor))
            else:
                return self.coarsen_v(var, factor)
    
        else:
            raise ValueError("Variable dimensions do not match any expected grid type.")




def interpolate_from_rho_to_u(field, method="additive"):

    """
    Interpolates the given field from rho points to u points.

    This function performs an interpolation from the rho grid (cell centers) to the u grid
    (cell edges in the xi direction). Depending on the chosen method, it either averages
    (additive) or multiplies (multiplicative) the field values between adjacent rho points
    along the xi dimension. It also handles the removal of unnecessary coordinate variables
    and updates the dimensions accordingly.

    Parameters
    ----------
    field : xr.DataArray
        The input data array on the rho grid to be interpolated. It is assumed to have a dimension
        named "xi_rho".

    method : str, optional, default='additive'
        The method to use for interpolation. Options are:
        - 'additive': Average the field values between adjacent rho points.
        - 'multiplicative': Multiply the field values between adjacent rho points. Appropriate for
          binary masks.

    Returns
    -------
    field_interpolated : xr.DataArray
        The interpolated data array on the u grid with the dimension "xi_u".
    """

    if method == "additive":
        field_interpolated = 0.5 * (field + field.shift(xi_rho=1)).isel(
            xi_rho=slice(1, None)
        )
    elif method == "multiplicative":
        field_interpolated = (field * field.shift(xi_rho=1)).isel(xi_rho=slice(1, None))
    else:
        raise NotImplementedError(f"Unsupported method '{method}' specified.")

    if "lat_rho" in field_interpolated.coords:
        field_interpolated.drop_vars(["lat_rho"])
    if "lon_rho" in field_interpolated.coords:
        field_interpolated.drop_vars(["lon_rho"])

    field_interpolated = field_interpolated.swap_dims({"xi_rho": "xi_u"})

    return field_interpolated


def interpolate_from_rho_to_v(field, method="additive"):

    """
    Interpolates the given field from rho points to v points.

    This function performs an interpolation from the rho grid (cell centers) to the v grid
    (cell edges in the eta direction). Depending on the chosen method, it either averages
    (additive) or multiplies (multiplicative) the field values between adjacent rho points
    along the eta dimension. It also handles the removal of unnecessary coordinate variables
    and updates the dimensions accordingly.

    Parameters
    ----------
    field : xr.DataArray
        The input data array on the rho grid to be interpolated. It is assumed to have a dimension
        named "eta_rho".

    method : str, optional, default='additive'
        The method to use for interpolation. Options are:
        - 'additive': Average the field values between adjacent rho points.
        - 'multiplicative': Multiply the field values between adjacent rho points. Appropriate for
          binary masks.

    Returns
    -------
    field_interpolated : xr.DataArray
        The interpolated data array on the v grid with the dimension "eta_v".
    """

    if method == "additive":
        field_interpolated = 0.5 * (field + field.shift(eta_rho=1)).isel(
            eta_rho=slice(1, None)
        )
    elif method == "multiplicative":
        field_interpolated = (field * field.shift(eta_rho=1)).isel(
            eta_rho=slice(1, None)
        )
    else:
        raise NotImplementedError(f"Unsupported method '{method}' specified.")

    if "lat_rho" in field_interpolated.coords:
        field_interpolated.drop_vars(["lat_rho"])
    if "lon_rho" in field_interpolated.coords:
        field_interpolated.drop_vars(["lon_rho"])

    field_interpolated = field_interpolated.swap_dims({"eta_rho": "eta_v"})

    return field_interpolated


def compute_cs(sigma, theta_s, theta_b):
    """
    Compute the S-coordinate stretching curves according to Shchepetkin and McWilliams (2009).

    Parameters
    ----------
    sigma : np.ndarray or float
        The sigma-coordinate values.
    theta_s : float
        The surface control parameter.
    theta_b : float
        The bottom control parameter.

    Returns
    -------
    C : np.ndarray or float
        The stretching curve values.

    Raises
    ------
    ValueError
        If theta_s or theta_b are not within the valid range.
    """
    if not (0 < theta_s <= 10):
        raise ValueError("theta_s must be between 0 and 10.")
    if not (0 < theta_b <= 4):
        raise ValueError("theta_b must be between 0 and 4.")

    C = (1 - np.cosh(theta_s * sigma)) / (np.cosh(theta_s) - 1)
    C = (np.exp(theta_b * C) - 1) / (1 - np.exp(-theta_b))

    return C


def sigma_stretch(theta_s, theta_b, N, type):
    """
    Compute sigma and stretching curves based on the type and parameters.

    Parameters
    ----------
    theta_s : float
        The surface control parameter.
    theta_b : float
        The bottom control parameter.
    N : int
        The number of vertical levels.
    type : str
        The type of sigma ('w' for vertical velocity points, 'r' for rho-points).

    Returns
    -------
    cs : xr.DataArray
        The stretching curve values.
    sigma : xr.DataArray
        The sigma-coordinate values.

    Raises
    ------
    ValueError
        If the type is not 'w' or 'r'.
    """
    if type == "w":
        k = xr.DataArray(np.arange(N + 1), dims="s_w")
        sigma = (k - N) / N
    elif type == "r":
        k = xr.DataArray(np.arange(1, N + 1), dims="s_rho")
        sigma = (k - N - 0.5) / N
    else:
        raise ValueError(
            "Type must be either 'w' for vertical velocity points or 'r' for rho-points."
        )

    cs = compute_cs(sigma, theta_s, theta_b)

    return cs, sigma


def compute_depth(zeta, h, hc, cs, sigma):
    """
    Compute the depth at different sigma levels.

    Parameters
    ----------
    zeta : xr.DataArray
        The sea surface height.
    h : xr.DataArray
        The depth of the sea bottom.
    hc : float
        The critical depth.
    cs : xr.DataArray
        The stretching curve values.
    sigma : xr.DataArray
        The sigma-coordinate values.

    Returns
    -------
    z : xr.DataArray
        The depth at different sigma levels.

    Raises
    ------
    ValueError
        If theta_s or theta_b are less than or equal to zero.
    """

    # Expand dimensions
    sigma = sigma.expand_dims(dim={"eta_rho": h.eta_rho, "xi_rho": h.xi_rho})
    cs = cs.expand_dims(dim={"eta_rho": h.eta_rho, "xi_rho": h.xi_rho})

    s = (hc * sigma + h * cs) / (hc + h)
    z = zeta + (zeta + h) * s

    return z
