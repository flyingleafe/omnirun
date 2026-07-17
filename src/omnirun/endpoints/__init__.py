"""Endpoint layer: process-wide sharing of physical-target state.

One *endpoint* is one physical target — an ssh login host or a provider HTTP
API. Several configured backends may point at the same endpoint (three slurm
partitions on one login node; several marketplace sections on one provider
account); everything that is per-target rather than per-backend (the ssh
session, the API rate-limit throttle, discovery facts) lives here so those
backends share instead of duplicating remote traffic.
"""

from omnirun.endpoints.manager import EndpointManager, Throttle

__all__ = ["EndpointManager", "Throttle"]
