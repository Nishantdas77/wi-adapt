# episodes/test1

Put the `.pcap` (Room A style) or `.csv` (Room B style) file(s) to predict in this folder, then run from the main project folder:

```bash
python src/run_episode.py episodes/test1
```

The predictions are written to `episodes/test1/predictions.csv` (`window_id, room, label, confidence`).
