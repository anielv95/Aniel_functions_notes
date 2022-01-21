def f_min_max(x,t1_n,t2_n):
    df_ = pd.DataFrame()
    df_['t'] = x[t1_n]
    df_ = pd.concat([df_[['t']],x.rename(columns={t2_n:'t'})[['t']]],ignore_index=True)
    return df_['t'].min(),df_['t'].max()

def f_create_yearmonth(x,t1_n,t2_n):
    df_ = pd.DataFrame()
    df_ = x.copy()
    tmin,tmax = f_min_max(x,t1_n,t2_n)
    deltayear = tmax.year-tmin.year
    date_aux = tmin
    if deltayear==0:
        num_month = tmin.month-tmax.month+1
    else:
        num_month = 12-tmin.month+1+(deltayear-1)*12+tmax.month
    for i in range(num_month):
        col = date_aux.year*100+date_aux.month
        date_aux = date_aux+datetime.timedelta(days=31)
        date_aux = date_aux.replace(day=1)
        df_[col] = col
    return df_.copy() ,num_month
