# coding: UTF-8

import os
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, make_response
from werkzeug.datastructures import FileStorage
from sklearn.preprocessing import LabelEncoder

# モデルの読み込み
with open('./static/model.pickle', mode='rb') as fp:
    model = pickle.load(fp)

app = Flask(__name__)


@app.route('/api/v1/pred', methods=['POST'])
def pred_csv():
    csv_data = request.files['data']
    # csvファイルのみ受け付ける
    if isinstance(csv_data, FileStorage) and csv_data.content_type == 'text/csv':
        original_df = pd.read_csv(csv_data)

        # 実際にサイトから応募する際に見れると思われるデータを選択する
        column_names = ["（派遣先）配属先部署　男女比　男", "勤務地　最寄駅2（駅名）", "（紹介予定）雇用形態備考", "（派遣先）配属先部署　人数", "勤務地　最寄駅1（分）",
                        "勤務地　最寄駅2（沿線名）", "勤務地　最寄駅1（駅名）", "給与/交通費　備考"]

        # スライシングで警告が出るので対策する
        processing_df = original_df[column_names].copy()

        # NaN に対する処理を行う
        label_encoder = LabelEncoder()
        for column_name in column_names:
            # 文字列に NaN が入っている場合は文字列の "NaN" を入れる
            if processing_df[column_name].dtype == object:
                processing_df[column_name] = processing_df[column_name].fillna("NaN")
                processing_df[column_name] = label_encoder.fit_transform(processing_df[column_name].values)

            # 数値に NaN が入っている場合 -1.0 を異常値とする（正規化はしない）
            else:
                processing_df[column_name] = processing_df[column_name].fillna(-1.0)

        # 前処理
        x_array = np.array(processing_df)

        # 推論を行う
        predict_data = model.predict(x_array)

        # Pandas 形式にして返す
        predict_df = pd.DataFrame(predict_data, columns=["応募数 合計"])
        marge_df = pd.concat([original_df, predict_df], axis=1, join="outer")

        return_csv = marge_df[["お仕事No.", "応募数 合計"]].to_csv()

        response = make_response(return_csv)
        response.headers["Content-Disposition"] = "attachment; filename=export.csv"
        response.headers['Content-Type'] = 'text/csv'

        return response

    else:
        return 'data is not csv'


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
