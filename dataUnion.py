import pandas as pd
import os

df = pd.read_csv('C:/Users/TIM/PycharmProjects/pythonTestPyTorch/IPC-image-data-master/runtimes.csv')
non_matching_count = 0
i = 0
num_rows = df.shape[0]
while i < num_rows:
    # i is task_row index
    task_name = df.iloc[i][0] + '-bolded-cs.png'
    fileExists = os.path.exists("C:/Users/TIM/PycharmProjects/pythonTestPyTorch/IPC-image-data-master/grounded/"
                                + task_name)
    if not fileExists:
        df.drop(df.index[i], axis=0, inplace=True)  # drop this row
        num_rows -= 1
        non_matching_count += 1
    print(fileExists)
    i += 1
print(non_matching_count)
df.to_csv('C:/Users/TIM/PycharmProjects/pythonTestPyTorch/IPC-image-data-master/runtimes.csv', index=False)
