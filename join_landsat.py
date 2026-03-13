import pandas as pd

landsat_list = [f'ls_station_{i}.csv' for i in range(0,162)]

final_ls = pd.DataFrame()

for files in landsat_list:
    files = pd.read_csv(files)
    if files.shape[1] > 44:
        print('greater than 44')
        files = files.iloc[:, :44]
    final_ls = pd.concat([final_ls, files],axis=0)

print(final_ls.shape)
final_ls.to_csv('landsat_train_feats.csv', index=False)