"""
Shared preprocessing utilities for environment-grouped PCA analyses.

Used by both fluxnet and ecosystem applications.
"""

import pandas as pd
import torch


def create_poolscale(df, columns, ddof=1):
    """
    Standardize columns on pooled (global) mean and std.

    Parameters
    ----------
    df : pd.DataFrame
        Data with feature columns.
    columns : list[str]
        Column names to standardize.
    ddof : int
        Degrees of freedom for std computation.

    Returns
    -------
    pd.DataFrame
        Standardized data (zero pooled mean, unit pooled std).
    """
    return (df[columns] - df[columns].mean()) / df[columns].std(ddof=ddof)


def create_envzeromean(df, columns, environments, ddof=0):
    """
    Center each environment group to zero mean, then scale by pooled std.

    Parameters
    ----------
    df : pd.DataFrame
        Data with feature columns.
    columns : list[str]
        Column names to process.
    environments : pd.Series
        Environment label for each row (same index as df).
    ddof : int
        Degrees of freedom for std computation.

    Returns
    -------
    pd.DataFrame
        Environment-centered, pool-scaled data.
    """
    env_col = '_env_temp_'
    result = df[columns].copy()
    result[env_col] = environments.values
    result[columns] = (
        result.groupby(env_col)[columns]
        .transform(lambda x: x - x.mean())
    )
    result.drop(columns=[env_col], inplace=True)
    result /= result.std(ddof=ddof)
    return result


def build_env_dicts(X_poolscale_df, X_envzeromean_df, environments):
    """
    Build per-environment torch tensor dictionaries.

    For poolscale: each environment slice is additionally centered to zero mean.
    For envzeromean: used as-is.

    Parameters
    ----------
    X_poolscale_df : pd.DataFrame
        Poolscale-standardized data.
    X_envzeromean_df : pd.DataFrame
        Environment-centered, pool-scaled data.
    environments : pd.Series
        Environment label per row.

    Returns
    -------
    env_dfs_poolscale : dict[str, torch.Tensor]
    env_dfs_envmeanzero : dict[str, torch.Tensor]
    """
    env_dfs_poolscale = {}
    env_dfs_envmeanzero = {}

    for env in environments.unique():
        mask = environments == env

        xe = X_poolscale_df[mask].values
        xe = xe - xe.mean(axis=0)
        env_dfs_poolscale[env] = torch.tensor(xe, dtype=torch.float32)

        xe = X_envzeromean_df[mask].values
        env_dfs_envmeanzero[env] = torch.tensor(xe, dtype=torch.float32)

    return env_dfs_poolscale, env_dfs_envmeanzero
