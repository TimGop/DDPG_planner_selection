import pandas as pd
import random


def import_data():
    df = pd.read_csv('IPC-image-data-master/runtimes.csv')
    df = df.drop(df.columns[list(range(18, 30, 1))], axis=1)
    return df  # create training test split not important how yet but lets get a decent first draft
    # return data


def batchSplit(df):
    df_best_times = (df.min(axis=1))
    number_of_rows = df.shape[0]
    training = []
    validation = []
    test = []
    current_task_batch = []
    for i in range(0, number_of_rows):
        current_row = df.iloc[i]
        current_row_name = df.iloc[i][0]
        current_row_name = current_row_name.split("-")[0]
        # skip unsolvable examples
        if df_best_times[i] > 1800:
            if (df.iloc[i][0].split("-")[0]) != (df.iloc[i+1][0].split("-")[0]):
                # append batch without current row
                x = random.random()
                if x <= 0.1:
                    for element in current_task_batch:
                        test.append(element)
                elif x <= 0.2:
                    for element in current_task_batch:
                        validation.append(element)
                else:
                    for element in current_task_batch:
                        training.append(element)
                current_task_batch = []
            continue
        if i + 1 == number_of_rows or current_row_name != df.iloc[i + 1][0].split("-")[0]:
            # iff next task not same type
            current_task_batch.append(current_row)
            # randomly append batch to training or test
            x = random.random()
            if x <= 0.1:
                for element in current_task_batch:
                    test.append(element)
            elif x <= 0.2:
                for element in current_task_batch:
                    validation.append(element)
            else:
                for element in current_task_batch:
                    training.append(element)
            current_task_batch = []  # empty current batch for next batch

        else:  # task is same type
            current_task_batch.append(current_row)
    df_train = pd.DataFrame(training)
    df_validation = pd.DataFrame(validation)
    df_test = pd.DataFrame(test)
    df_train.to_csv(r'IPC-image-data-master/problem_splits/training.csv',
                    index=False)
    df_validation.to_csv(r'IPC-image-data-master/problem_splits'
                         r'/validation.csv', index=False)
    df_test.to_csv(r'IPC-image-data-master/problem_splits/testing.csv',
                   index=False)


def randSplit(df):
    number_of_rows = df.shape[0]
    training = []
    test = []
    validation = []
    for i in range(0, number_of_rows):
        current_row = df.iloc[i]
        r = random.random()
        if r <= 0.1:
            training.append(current_row)
        elif r <= 0.2:
            validation.append(current_row)
        else:
            test.append(current_row)
    df_train = pd.DataFrame(training)
    df_validation = pd.DataFrame(validation)
    df_test = pd.DataFrame(test)
    df_train.to_csv(r'IPC-image-data-master/problem_splits/training.csv',
                    index=False)
    df_validation.to_csv(r'IPC-image-data-master/problem_splits'
                         r'/validation.csv', index=False)
    df_test.to_csv(r'IPC-image-data-master/problem_splits/testing.csv',
                   index=False)


def main():
    df = import_data()
    batchSplit(df)
    # randSplit(df)


if __name__ == "__main__":
    main()
