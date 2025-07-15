from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__, static_folder='templates')
app = application


@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    else:
        try:
            data = CustomData(
                cc_num=float(request.form.get('cc_num')),
                category=request.form.get('category'),
                first=request.form.get('first'),
                last=request.form.get('last'),
                gender=request.form.get('gender'),
                job=request.form.get('job'),
                dob=request.form.get('dob'),
                date=request.form.get('date'),
                time=request.form.get('time')
            )

            new_data = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(new_data)

            result_label = "Fraudulent Transaction"
            if pred[0] == 1:
                result_label = "Fraudulent Transaction"
            else:
                result_label = "Legitimate Transaction"

            return render_template("form.html", final_result=result_label)

        except Exception as e:
            return render_template("form.html",
                                   final_result=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
