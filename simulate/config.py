from dataclasses import dataclass


@dataclass
class InjectionConfig:
    seed: int = 42

    # how many rows to inject per single spike event
    n_extra_fraud: int = 150
    n_extra_demand: int = 150

    # how many time steps get a fraud/demand event injected
    # (bumped up so the model sees more examples of each class)
    num_fraud_steps: int = 80
    num_demand_steps: int = 80

    # how many DIFFERENT fraud IPs / BINs get reused across a fraud event
    # small number = looks like a concentrated attack from one source
    n_fraud_ips: int = 5
    n_fraud_bins: int = 6


CONFIG = InjectionConfig()