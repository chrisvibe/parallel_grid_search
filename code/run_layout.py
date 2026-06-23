import dataclasses


@dataclasses.dataclass(frozen=True)
class RunLayout:
    """File and directory names for a grid search output directory.

    All paths are relative to the run root (output_path).  The data sub-directory
    holds per-batch Parquets, state.db, and the compacted result file.

    Usage::

        data_dir   = output_path / RUN.data_dir
        state_path = data_dir / RUN.state_db
        compacted  = data_dir / RUN.compacted_file
        flag       = output_path / RUN.completed_flag
    """
    data_dir:       str = 'data'                        # folder holding batch Parquets + state.db
    compacted_file: str = 'data.parquet'                # merged results written at completion
    state_db:       str = 'state.db'                    # ephemeral SQLite job-queue; all SQLiteLocks share this file
    building_lock:  str = 'state.building.lock'         # file-based init lock (used before state.db exists)
    completed_flag: str = 'GRID_SEARCH_COMPLETED.flag'  # plain file written when all jobs done; not a lock


RUN = RunLayout()
