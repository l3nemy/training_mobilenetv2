

import os
import pandas as pd


deleted_class_id = [0, 2, 4] + \
    list(range(5, 14)) + list(range(15, 17)) + list(range(18, 33)) + \
    list(range(38, 40)) + [41, 42]


def main():
    for ds in ['Test', 'Train']:
        df = pd.read_csv(f'./datasets/{ds}.csv')

        for i, row in df.iterrows():
            if row["ClassId"] in deleted_class_id:
                try:
                    os.remove(os.path.join('./datasets', row['Path']))
                except:
                    pass
                finally:
                    df.drop(index=i, inplace=True)
    df.to_csv(f'./out_{ds}.csv')
    for d in deleted_class_id:
        try:
            os.removedirs(f'./datasets/Train/{d}')
        except:
            pass


if __name__ == '__main__':
    main()
