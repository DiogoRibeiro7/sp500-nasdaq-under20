# sp500-nasdaq-under20

Helper scripts that pull the latest S&P 500 + NASDAQ-100 components, find
those that recently traded below a target price, and download a year of OHLCV
history for each of them.

## Usage

The price filter now defaults to **30 USD**, which roughly doubles the number
of symbols captured compared to the earlier 20 USD cap. Pass `--max-price`
when running a script if you want to use some other ceiling.

### Download one CSV per ticker

```bash
python scripts/under20_stocks.py --max-price 35
```

This writes each ticker's history under `data_under_20/`.

### Update the consolidated CSV

```bash
python scripts/update_under20_master_csv.py --max-price 35
```

This keeps `data/under20_history.csv` up to date by appending any missing rows.
