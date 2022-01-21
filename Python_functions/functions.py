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

def f_to_date(x #dataframe
              ,l_names #list of names to convert in datetime type
              , l_format #list of formats
             ):#l_names could be a list or string, l_format could be a str or a list 
    #as big as the list l_names
    y = pd.DataFrame()
    y = x.copy()
    if type(l_names)==type(['hi']):
        if type(l_format)==type(['hi']):
            if len(l_format)==len(l_names):
                for i in range(len(l_names)):
                    y[l_names[i]] = pd.to_datetime(y[l_names[i]], format = l_format[i])
            else:
                print('l_format should be a list as long as l_names')
        else:
            if type(l_format) == type('hi'):
                for i in range(len(l_names)):
                    y[l_names[i]] = pd.to_datetime(y[l_names[i]], format = l_format)
            else:
                print('l_format should be a list as long as l_names or a str') 
    else:
        if type(l_names)==type('hi'):
            if type(l_format)==type('hi'):
                y[l_names] = pd.to_datetime(y[l_names],format=l_format)
            else:
                print('l_names is a str then l_format should be a str')
        else:
            print('l_names should be a list or a string')
    return y.copy()

def build_model_v2_6(hp):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(103,)))
    # Tune the number of layers.
    for i in range(
        #hp.Int("num_layers", 1, 12)
        1
    ):
        model.add(
            layers.Dense(
                # Tune number of units separately.
                #units=hp.Int("units", min_value=1312-32, max_value=1312+32, step=32),
                units=32,
                #activation=hp.Choice("activation", ["tanh",'sigmoid']),
                activation="tanh",
                #activation="tanh",
                kernel_initializer=keras.initializers.RandomNormal(stddev=np.sqrt(2/(103+32))),
            )
        )
    #if hp.Boolean("dropout"):
    #    model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(units = 401,
                           activation='softplus',
                           kernel_initializer=keras.initializers.RandomNormal(stddev=np.sqrt(2/(32+401)))
                          )
             )
    learning_rate = hp.Float("lr", min_value=1e-3, max_value=1e2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mape",'mae','mse'],
    )
    return model

build_model_v2_6(kt.HyperParameters())

tuner_v2_6 = kt.RandomSearch(
    hypermodel=build_model_v2_6,
    objective="val_mse",
    max_trials=10,
    executions_per_trial=1,
    overwrite=True,
    directory="my_dir_v2",
    project_name="softplus-initialization-0-32",
    seed=42,
)

tuner_v2_6.search(X_train_cat_temp, y_train_prepared, epochs=100, validation_data=(X_val_cat_temp, y_val_prepared))

tuner_v2_6.results_summary()

# Get the top 2 models.
models_6 = tuner_v2_6.get_best_models(num_models=2).copy()
best_model_6 = models_6[0]
# Build the model.
# Needed for `Sequential` without specified `input_shape`.
#best_model.build(input_shape=(None, 28, 28))
best_model_6.summary()

best_model_6.evaluate(X_val_cat_temp,y_val_prepared)

y_predicted = best_model_6.predict(X_val_cat_temp)
