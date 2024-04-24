import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from typing import List
import datetime as dt
import statsmodels.api as sm


curr_dir = os.path.dirname(os.path.abspath(__file__))


def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    return fig


def simple_linear_regression(x, y):
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    return results


class DataStore:

    data_file = os.path.join(curr_dir, '../data/fx_data.csv')
    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    fx_g10 = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK']

    @classmethod
    def get_fx_spots(cls, fx_pairs: List[str], start_date: dt.date, end_date: dt.date):
        df = cls.data.loc[start_date:end_date, [f'{fx} Curncy' for fx in fx_pairs]].rename(columns={f'{fx} Curncy': fx for fx in fx_pairs})
        return df.ffill()

    @classmethod
    def get_first_date(cls):
        return cls.data.index[0].date()

    @classmethod
    def get_last_date(cls):
        return cls.data.index[-1].date()

    @classmethod
    def get_fx_spot_matrix(cls, fx_pairs: List[str], date: dt.date, base_ccy='USD'):
        if not all([base_ccy in fx for fx in fx_pairs]):
            raise KeyError(f'Not all fx pair contains base ccy {base_ccy}')

        df = cls.get_fx_spots(fx_pairs=fx_pairs, start_date=cls.data.index[0].date(), end_date=date)
        df = df.iloc[-1].to_frame().T
        d = df.index[0].date()
        # change df columns to base_ccy as ccy1 in ccy1ccy2 pair
        new_columns = []
        for col in df.columns:
            if col[:3] != base_ccy:
                new_columns.append(col[3:] + col[:3])
                df[col] = 1 / df[col]
            else:
                new_columns.append(col)
        df.columns = new_columns
        vals = df.to_dict('records')[0]

        columns = [base_ccy] + [fx[3:] if fx[:3]==base_ccy else fx[:3] for fx in fx_pairs]
        index = columns[-1::-1]
        m = pd.DataFrame(np.zeros([len(columns), len(columns)]), columns=columns, index=index)
        for ccy2, item in m.iterrows():
            for ccy1 in item.index:
                if ccy1 == ccy2:
                    m.loc[ccy2, ccy1] = np.nan
                    continue
                fx = ccy1 + ccy2
                if base_ccy == ccy1:
                    val = vals[fx]
                elif base_ccy == ccy2:
                    val = 1 / vals[ccy2+ccy1]
                else:
                    val = vals[base_ccy + ccy2] / vals[base_ccy + ccy1]
                m.loc[ccy2, ccy1] = val
        return m, d

    @classmethod
    def get_fx_spot_pct_change_matrix(cls, fx_pairs: List[str], date: dt.date, base_ccy='USD'):
        m0, d = cls.get_fx_spot_matrix(fx_pairs=fx_pairs, date=date, base_ccy=base_ccy)
        d1 = d - dt.timedelta(days=1)
        m1, d1 = cls.get_fx_spot_matrix(fx_pairs=fx_pairs, date=d1, base_ccy=base_ccy)
        df = (m0 - m1) / m1
        return df, d







