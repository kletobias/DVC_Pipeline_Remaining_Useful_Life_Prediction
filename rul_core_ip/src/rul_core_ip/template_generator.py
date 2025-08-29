# dependencies/templates/generate_dvc_yaml_core.py
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import jinja2

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from omegaconf import DictConfig


def generate_dvc_yaml_core(
    stages_list: list,
    search_path: str,
    template_name: str,
    dvc_yaml_file_path: str,
    plots_list: list | None = None,
) -> None:
    """
    Render dvc.yaml from Jinja template and Hydra-built stage list.
    Whitespace-stripping flags stop hidden TAB / U+2502 issues that
    break DVCs YAML schema.
    """
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(searchpath=search_path),
        autoescape=False,
        trim_blocks=False,
        lstrip_blocks=False,
        keep_trailing_newline=True,
    )
    template = env.get_template(template_name)
    rendered = template.render(stages=stages_list, plots=plots_list)
    rendered = re.sub(r"\s$", "", rendered, flags=re.MULTILINE)
    with open(dvc_yaml_file_path, "w", encoding="utf-8") as f:
        f.write(rendered)


if __name__ == "__main__":
    import sys

    import hydra

    sys.argv = [sys.argv[0], "pipeline=orchestrate_dvc_flow"]

    @hydra.main(version_base=None, config_path="../../configs", config_name="config")
    def main(cfg: DictConfig):
        generate_dvc_yaml_core(
            stages_list=cfg.pipeline.stages,
            search_path=cfg.pipeline.search_path,
            template_name=cfg.pipeline.template_name,
            dvc_yaml_file_path=cfg.pipeline.dvc_yaml_file_path,
            plots_list=cfg.pipeline.get("plots"),
        )

    main()
