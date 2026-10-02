# Dubach option data: local inventory and recovery search (2026-10-01)

Read-only search; nothing was downloaded, moved, extracted or modified. This folder holds the only files written.

## Reference list

The documented list is the **104 symbols** in the README of the repository clone (`data/raw/github_options_repo/README.md`). The identical list is hard-coded in that clone's `download.py` and `download.sh`; all three copies agree, with 104 unique symbols. The repository's own layout is `<ticker>/options.parquet` and `<ticker>/underlying.parquet`; its quick-download example saves `<ticker>_options.parquet`. Both naming patterns were searched.

## Local inventory: `data/raw/github_options/`

- **Present:** 60 of 104 tickers, each with both `*_options.parquet` and `*_underlying.parquet`. No local file is outside the list. Together 6.2 GB and 285,071,100 option rows.
- **Per-ticker detail:** size, rows, first date and last date are in `present_dubach_tickers.csv`. Most start on 2008-01-02. The later starts are ABBV 2013-01-02, AVGO 2009-08-24, GOOG 2010-01-04, META 2021-07-08, NEE 2010-06-23, QQQ 2011-03-23, TSLA 2010-07-08 and V 2008-03-28. All end between 2025-12-12 and 2025-12-22.
- **Missing:** 44 tickers, listed in `missing_dubach_tickers.txt`: abt acn aig bk bkng bmy brk.b cl cof cvs dhr emr fdx gd gild gm googl ibm intu isrg lin mdlz mdt met mmm mo now pltr pm pypl qcom rtx sbux schw spg tgt tmus txn uber unp ups usb vix vz.

## Where copies were searched

| Location | Result |
|---|---|
| Spotlight index, whole disk (`*_options.parquet`, `options.parquet`, `underlying.parquet`) | 360 and 180 hits, **all** inside three copies of this project (this folder, `~/Desktop/options_research 2`, the BIR snapshot clone): the same 60 tickers |
| Full filesystem search of `~` including hidden folders and `~/Library` (Mail, Messages, Safari and app containers excluded) | no option/underlying parquet outside the project copies; the only other parquet files are unrelated (`~/Desktop/benchmarkzoo`) |
| `~/Desktop/options_research_full.zip` (10.4 GB, 2026-06-07), contents listed without extracting | a full archive of this project: the **same 60 tickers**, none of the 44 |
| 737 other archives outside `~/Library` (LEAN sample data, other projects) and 82 under `~/Library` (app and Homebrew caches), contents listed | none contains a Dubach option or underlying parquet |
| `~/Downloads`, `~/Documents` | none (personal documents only, not opened) |
| `/private/tmp`, `/private/var/folders`, `/Users/Shared`, `/opt` | none |
| External volumes (`/Volumes`) | none mounted (only the system disk) |
| iCloud Drive / `~/Library/CloudStorage` | not present on this machine |
| Old project path `~/Desktop/guided-research` | no longer exists; its Claude Code transcript folder is empty |
| Shell histories (`~/.zsh_history`, `~/.bash_history`, `~/.zsh_sessions/*`), `~/.python_history` | no line mentions the dataset, its URL or its download scripts |
| IPython history | none exists |
| macOS download records (quarantine database, 442 records) | none from `philippdubach` or the dataset |
| Claude Code transcripts (`~/.claude/projects`) | only this session mentions the dataset URL. The project's transcripts name 76 distinct `*_options/underlying.parquet` strings, **none** of them one of the 44 missing tickers |
| Project download scripts (`scripts/download_new_tickers.py`, `.sh`) | target exactly the 60-ticker universe; nothing was ever requested for the 44 |
| Time Machine | no backup destination configured; only macOS-update APFS snapshots exist (not mounted; that needs admin access) |

**Could not be searched:**
- `~/.Trash`: macOS denies this terminal access ("Operation not permitted"). Check it in Finder, or grant the terminal Full Disk Access.
- macOS-update snapshots: not mounted.
- Storage not attached to this machine: external disks, other computers, cloud accounts.

## Conclusion

Nothing beyond the 60 present tickers is recoverable from this machine. Everything found points one way: the 44 missing tickers were never downloaded here. The project's scripts requested only the 60, and no history, transcript or download record mentions any of the 44.
