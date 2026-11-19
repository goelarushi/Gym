# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A merged config dict larger than one environment string may hold goes to the servers through a file."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pytest import MonkeyPatch

import nemo_gym.config.global_config as global_config_module
from nemo_gym.config.global_config import (
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_CONFIG_DICT_FILE_ENV_VAR_NAME,
    NEMO_GYM_LOG_DIR_KEY_NAME,
    maybe_get_global_config_dict,
)
from nemo_gym.server_runtime.supervisor import server_supervisor
from nemo_gym.utils.server_process import _has_injected_global_config_env


class TestConfigDictFile:
    """Linux refuses a single argument or environment string above MAX_ARG_STRLEN (128 KiB): a run config naming
    ~1,200 Harbor tasks made every server spawn fail with ``Argument list too long``. Above a threshold under that
    limit the supervisor writes the YAML to a file and names it in ``NEMO_GYM_CONFIG_DICT_FILE``; a server reads
    the file where it would read ``NEMO_GYM_CONFIG_DICT``."""

    def test_threshold_leaves_room_under_the_kernel_limit(self) -> None:
        assert server_supervisor._CONFIG_DICT_ENV_MAX_BYTES < 131072

    def test_the_file_lands_in_the_log_dir_when_one_is_configured(self, tmp_path: Path) -> None:
        yaml_str = OmegaConf.to_yaml(DictConfig({"a": {"b": list(range(50))}}))
        path = server_supervisor._write_config_dict_file(
            yaml_str, DictConfig({NEMO_GYM_LOG_DIR_KEY_NAME: str(tmp_path / "logs")})
        )
        assert path.parent == tmp_path / "logs" and path.name.startswith("nemo_gym_config_dict_")
        assert path.read_text(encoding="utf-8") == yaml_str

    def test_without_a_log_dir_the_file_goes_to_tmp(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        monkeypatch.setattr(server_supervisor.tempfile, "tempdir", None)
        path = server_supervisor._write_config_dict_file("k: v\n", DictConfig({}))
        assert path.parent == tmp_path and path.read_text(encoding="utf-8") == "k: v\n"

    def test_a_server_reads_the_config_dict_from_the_file(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("a:\n  b: 1\n", encoding="utf-8")
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setenv(NEMO_GYM_CONFIG_DICT_FILE_ENV_VAR_NAME, str(config_file))
        monkeypatch.setattr(global_config_module, "_GLOBAL_CONFIG_DICT", None)

        assert _has_injected_global_config_env()
        config = maybe_get_global_config_dict()

        assert config is not None and config["a"]["b"] == 1
        # Cached like the environment-string path.
        assert maybe_get_global_config_dict() is config

    def test_the_environment_string_wins_when_both_are_set(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("a: file\n", encoding="utf-8")
        monkeypatch.setenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, "a: env\n")
        monkeypatch.setenv(NEMO_GYM_CONFIG_DICT_FILE_ENV_VAR_NAME, str(config_file))
        monkeypatch.setattr(global_config_module, "_GLOBAL_CONFIG_DICT", None)

        config = maybe_get_global_config_dict()

        assert config is not None and config["a"] == "env"
        monkeypatch.setattr(global_config_module, "_GLOBAL_CONFIG_DICT", None)

    def test_a_bare_process_is_not_a_server(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_FILE_ENV_VAR_NAME, raising=False)
        assert not _has_injected_global_config_env()
