# dependencies/config_schemas/RootConfig.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from dependencies.modeling.ridge_optuna_trial import RidgeOptunaTrialConfig
from dependencies.tests.check_required_columns import CheckRequiredColumnsConfig
from dependencies.tests.check_row_count import CheckRowCountConfig
from dependencies.transformations.ingest_data import IngestDataConfig
from dependencies.transformations.remove_constant_and_high_corr_features import (
    RemoveConstantAndHighCorrFeaturesConfig,
)


@dataclass
class IOPolicyConfig:
    READ_INPUT: bool = True
    WRITE_OUTPUT: bool = True


@dataclass
class DataVersionsConfig:
    name: str = MISSING
    data_version_input: str = MISSING
    data_version_output: str = MISSING
    description: str = MISSING
    dataset_url: str = MISSING
    data_version: str = MISSING


@dataclass
class HydraConfig:
    job: dict[str, str] = field(default_factory=dict)
    run: dict[str, str] = field(default_factory=dict)
    sweep: dict[str, str] = field(default_factory=dict)


@dataclass
class LogCfgJobConfig:
    log_for_each_step: bool = False
    output_cfg_job_directory_path: str = MISSING
    output_cfg_job_file_path: str = MISSING
    resolve: bool = True


@dataclass
class LoggingUtilsConfig:
    log_directory_path: str = MISSING
    log_file_path: str = MISSING
    formatter: str = "%(asctime)s %(levelname)s:%(message)s"
    level: int = 20
    log_cfg_job: LogCfgJobConfig = field(default_factory=LogCfgJobConfig)


@dataclass
class SetupConfig:
    script_base_name: str = ""


@dataclass
class MLExperimentsConfig:
    rng_seed: int = MISSING
    n_jobs_study: int = MISSING
    n_jobs_final_model: int = MISSING
    target_col_modeling: str = MISSING
    year_col: str = MISSING
    train_range: list[int] = field(default_factory=list)
    val_range: list[int] = field(default_factory=list)
    test_range: list[int] = field(default_factory=list)
    experiment_prefix: str = MISSING
    experiment_id: str = MISSING
    artifact_directory_path: str = MISSING
    permutation_importances_filename: str = MISSING
    randomforest_importances_filename: str = MISSING
    top_n_importances: int = MISSING
    optuna_n_trials: int = MISSING
    direction: str = MISSING
    scoring: dict[str, str] = field(default_factory=dict)


@dataclass
class PathsDirectoriesConfig:
    project_root: str = MISSING
    bin: str = MISSING
    configs: str = MISSING
    data: str = MISSING
    dependencies: str = MISSING
    documentation: str = MISSING
    logs: str = MISSING
    outputs: str = MISSING
    scripts: str = MISSING
    templates: str = MISSING


@dataclass
class PathsConfig:
    directories: PathsDirectoriesConfig | None = field(
        default_factory=PathsDirectoriesConfig
    )


@dataclass
class ColumnNamesConfig:
    unit_col_name: str = "unit"


@dataclass
class TransformationsConfig:
    """
    You can add more typed configs if you have multiple transformations.
    Use 'RETURNS' to specify if a step returns a DataFrame or None.
    """

    check_required_columns: bool | None
    check_row_count: bool | None
    return_type: str | None
    column_names: ColumnNamesConfig | None
    name: str | None
    ingest_data: IngestDataConfig | None
    drop_constant_and_high_corr_features: RemoveConstantAndHighCorrFeaturesConfig | None
    ridge_optuna_trial: RidgeOptunaTrialConfig | None


@dataclass
class UtilityFunctionReadConfig:
    """Parameters for reading Parquet."""

    input_file_path: str
    low_memory: bool


@dataclass
class UtilityFunctionWriteConfig:
    """Parameters for writing to Parquet."""

    output_file_path: str
    include_index: bool


@dataclass
class UtilityFunctionMetadataConfig:
    """Parameters for generating and saving metadata."""

    data_file_path: str
    output_metadata_file_path: str


@dataclass
class UtilityFunctionsConfig:
    """
    Combines read, write, and metadata config under a single Hydra group,
    to keep them together but still allow dictionary unpacking in universal_step.
    """

    utility_function_read: UtilityFunctionReadConfig
    utility_function_write: UtilityFunctionWriteConfig
    utility_function_metadata: UtilityFunctionMetadataConfig


@dataclass
class DataStorageConfig:
    # Number of fields: 10
    split: str
    suffix: str
    input_file_extension: str
    output_file_extension: str
    input_metadata_file_extension: str
    output_metadata_file_extension: str
    input_params_file_extension: str
    output_params_file_extension: str
    input_params_file_path: str
    output_params_file_path: str
    input_file_path: str
    input_metadata_file_path: str
    output_file_path: str
    output_metadata_file_path: str
    run_id_outputs_directory_path: str


@dataclass
class StageConfig:
    name: str
    cmd_python: str | None
    script: str | None
    overrides: dict[str, Any] | None
    desc: str | None
    frozen: bool | None
    deps: list[str] | None
    outs: list[str] | None


@dataclass
class PlotConfig:
    # Number of fields: 3
    template: str | None = None
    x: str | None = None
    y: str | None = None


@dataclass
class Pipeline:
    stages: list[StageConfig] | None = field(default_factory=list)
    plots: list[PlotConfig] | None = field(default_factory=list)
    stages_to_run: list[str] | None = field(default_factory=list)
    force_run: bool | None = None
    pipeline_run: bool | None = None
    allow_dvc_changes: bool | None = None
    skip_generation: bool | None = None
    search_path: str | None = None
    template_name: str | None = None
    dvc_yaml_file_path: str | None = None
    log_file_path: str | None = None


@dataclass
class TestsConfig:
    check_required_columns: CheckRequiredColumnsConfig = field(
        default_factory=CheckRequiredColumnsConfig
    )
    check_row_count: CheckRowCountConfig = field(default_factory=CheckRowCountConfig)


@dataclass
class TestParamsConfig:
    # Number of fields: 2
    required_columns: list[str] | None = field(default_factory=list)
    row_count: int = 0


@dataclass
class RootConfig:
    data_storage: DataStorageConfig
    transformations: TransformationsConfig
    tests: TestsConfig
    cmd_python: str
    universal_step_script: str
    dvc_default_desc: str
    rng_seed: int
    data_versions: DataVersionsConfig
    hydra: HydraConfig
    logging_utils: LoggingUtilsConfig
    ml_experiments: MLExperimentsConfig
    paths: PathsConfig
    setup: SetupConfig
    pipeline: Pipeline
    io_policy: IOPolicyConfig
    utility_functions: UtilityFunctionsConfig
    test_params: TestParamsConfig


cs = ConfigStore.instance()


cs.store(group="setup", name="base_schema", node=SetupConfig)
cs.store(group="logging_utils", name="base_schema", node=LoggingUtilsConfig)
cs.store(group="io_policy", name="base_schema", node=IOPolicyConfig)
cs.store(group="transformations", name="select_schema", node=TransformationsConfig)
cs.store(group="utility_functions", name="base_schema", node=UtilityFunctionsConfig)
cs.store(group="data_storage", name="base_schema", node=DataStorageConfig)
cs.store(group="tests", name="base_schema", node=TestsConfig)
cs.store(group="test_params", name="base_schema", node=TestParamsConfig)
cs.store(group="data_versions", name="base_schema", node=DataVersionsConfig)
cs.store(group="hydra", name="default_schema", node=HydraConfig)
cs.store(group="logging_utils", name="default_schema", node=LoggingUtilsConfig)
cs.store(group="ml_experiments", name="base_schema", node=MLExperimentsConfig)
cs.store(group="paths", name="default_schema", node=PathsConfig)
cs.store(group="setup", name="base_schema", node=SetupConfig)
cs.store(group="pipeline", name="base_schema", node=Pipeline)

cs.store(name="root_config", node=RootConfig)
