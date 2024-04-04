from src.FlightPricePrediction.pipelines.prediction_pipeline import CustomData,PredictPipeline

from flask import Flask,request,render_template,jsonify


app=Flask(__name__) 


@app.route('/')
def home_page():
    return render_template("index.html")


@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    
    else:
        data=CustomData(
            
            
            Airline = request.form.get('Airline'),
            Source= request.form.get('Source'),
            Destination = request.form.get('Destination'),
            Total_Stops = request.form.get('Total_Stops'),
            Month_of_Journey= request.form.get('Month_of_Journey'),
            Day_of_Journey = request.form.get('Day_of_Journey'),
            Duration_in_minute=float(request.form.get('Duration_in_minute'))
        )
        # this is my final data
        final_data=data.get_data_as_dataframe()
        
        predict_pipeline=PredictPipeline()
        
        pred=predict_pipeline.predict(final_data)
        
        result=round(pred[0],2)
        
        return render_template("result.html",final_result=result)

#execution begin
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=80)