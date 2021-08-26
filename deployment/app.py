from os import link
import flask
from flask.globals import request
from flask import Flask, render_template
# library used for prediction
import numpy as np
import pandas as pd
import pickle
# library used for insights
import json
import plotly
import plotly.express as px

app = Flask(__name__, template_folder = 'templates')

link_active = None
# render home template
@app.route('/')
def main():
    return(render_template('home.html', title = 'Home'))

# load pickle file
model = pickle.load(open('model/rf_classifier.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

@app.route('/form')
def form():
    show_prediction = False
    link_active = 'Form'
    return(render_template('form.html', title = 'Form', show_prediction = show_prediction, link_active = link_active))

@app.route('/insights')
def insights():
    link_active = 'Insights'

    df = pd.read_csv('online_shoppers_intention.csv')
    df['Revenue'] = np.where(df['Revenue'] == True, 'Yes', 'No')
    df.rename(columns={'Revenue':'Intention to Buy'}, inplace = True)
    color_map = {'Yes': '#FFBF00', 'No': '#36454F'}

    df_sorted = df.sort_values('Intention to Buy', ascending = True)
    fig1 = px.scatter(
            df_sorted, x = 'BounceRates', y='ExitRates',
            color='Intention to Buy', color_discrete_map=color_map,
            labels = {
                "BounceRates": "Bounce Rates", "ExitRates" : "Exit Rates"
            }
        )
    fig1.update_layout(legend_traceorder='reversed')
    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    fig2 = px.box(
            df, x = 'Intention to Buy', y='PageValues', color='Intention to Buy',
            color_discrete_map=color_map,
            labels = {
                "PageValues" : "Page Values"
            }
        )
    fig2.update_layout(legend_traceorder='reversed')
    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    dist_vt = df.groupby(['VisitorType', "Intention to Buy"]).count()[["Administrative"]]
    cat_group = df.groupby(['VisitorType']).count()[["Administrative"]]
    dist_vt["percentage"] = dist_vt.div(cat_group, level = 'VisitorType') * 100
    dist_vt.reset_index(inplace = True)
    dist_vt.columns = ['VisitorType', "Intention to Buy", "count", "percentage"]
    dist_vt = dist_vt.sort_values(['VisitorType', 'Intention to Buy'], ascending=True)
    dist_vt['VisitorType'] = np.where(
        dist_vt['VisitorType'] == 'Returning_Visitor', 'Returning Visitor',
        np.where(dist_vt['VisitorType'] == 'New_Visitor',  'New Visitor', 'Other')
    )
    fig3 = px.bar(
            dist_vt, x = 'VisitorType', y = 'count', color = 'Intention to Buy', barmode="group",
            color_discrete_map=color_map,
            labels = {
                "VisitorType" : "Visitor Type"
            }
        )
    fig3.update_layout(showlegend=False)
    graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    fig4 = px.bar(
            dist_vt, x = 'VisitorType', y = 'percentage', color = 'Intention to Buy', barmode="group",
            color_discrete_map=color_map, range_y = [0, 100],
            labels = {
                "VisitorType" : "Visitor Type"
            }
        )
    fig4.update_layout(showlegend=False)
    graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
    
    df['Weekend'] = np.where(df['Weekend'] == True, 'Yes', 'No')
    dist_weekend = df.groupby(['Intention to Buy', "Weekend"]).count()[["Administrative"]]
    cat_group2 = df.groupby(['Weekend']).count()[["Administrative"]]
    dist_weekend["percentage"] = dist_weekend.div(cat_group2, level = 'Weekend') * 100
    dist_weekend.reset_index(inplace = True)
    dist_weekend.columns = ["Intention to Buy", 'Weekend', "count", "percentage"]
    fig5 = px.bar(
            dist_weekend, x = 'Weekend', y = 'percentage', color = 'Intention to Buy', barmode="group",
            color_discrete_map=color_map, range_y = [0, 100],
        )
    fig5.update_layout(showlegend=False)
    graph5JSON = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)

    dist_vt_weekend = df[df['VisitorType'] == 'New_Visitor'].groupby(['Intention to Buy', "Weekend"]).count()[["Administrative"]]
    cat_group3 = df[df['VisitorType'] == 'New_Visitor'].groupby(['Weekend']).count()[["Administrative"]]
    dist_vt_weekend["percentage"] = dist_vt_weekend.div(cat_group3, level = 'Weekend') * 100
    dist_vt_weekend.reset_index(inplace = True)
    dist_vt_weekend.columns = ["Intention to Buy", 'Weekend', "count", "percentage"]
    fig6 = px.bar(
            dist_vt_weekend, x = 'Weekend', y = 'percentage', color = 'Intention to Buy', barmode="group",
            color_discrete_map=color_map, range_y = [0, 100],
        )
    fig6.update_layout(showlegend=False)
    graph6JSON = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)

    return(render_template('insights.html', title = 'Insights', link_active = link_active, graph1JSON = graph1JSON, graph2JSON = graph2JSON, graph3JSON = graph3JSON, graph4JSON = graph4JSON, graph5JSON = graph5JSON, graph6JSON = graph6JSON))

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering prediction result.
    '''
    link_active = 'Result'
    show_prediction = True

    # retrieve data
    Administrative = int(request.form.get('Administrative'))
    Administrative_Duration = float(request.form.get('Administrative_Duration'))
    ProductRelated = int(request.form.get('ProductRelated'))
    ProductRelated_Duration = float(request.form.get('ProductRelated_Duration'))
    BounceRates = float(request.form.get('BounceRates'))
    ExitRates = float(request.form.get('ExitRates'))
    PageValues = float(request.form.get('PageValues'))
    Month = int(request.form.get('Month'))
    SpecialDay = request.form.get('SpecialDay')
    Weekend = request.form.get('Weekend')
    VisitorType = request.form.get('VisitorType')
    TrafficType = request.form.get('TrafficType')
    OperatingSystems = request.form.get('OperatingSystems')
    Browser = request.form.get('Browser')
    Region = request.form.get('Region')

    # transform to log
    Administrative = np.log1p(Administrative)
    Administrative_Duration = np.log1p(Administrative_Duration)
    ProductRelated = np.log1p(ProductRelated)
    ProductRelated_Duration = np.log1p(ProductRelated_Duration)
    BounceRates = np.log1p(BounceRates)
    ExitRates = np.log1p(ExitRates)
    PageValues = np.log1p(PageValues)

    # set previously known values for one-hot encoding
    known_SpecialDay = [0, 1]
    known_OperatingSystems = [1, 2, 3, 'other']
    known_Browser = [1, 2, 'other']
    known_Region = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    known_VisitorType = ['New_Visitor', 'Other', 'Returning_Visitor']
    known_Weekend = [False, True]

    # encode the categorical value
    SpecialDay_type = pd.Series([SpecialDay])
    SpecialDay_type = pd.Categorical(SpecialDay_type, categories = known_SpecialDay)
    SpecialDay_input = pd.get_dummies(SpecialDay_type, prefix = 'SpecialDay', drop_first=True)

    OperatingSystems_type = pd.Series([OperatingSystems])
    OperatingSystems_type = pd.Categorical(OperatingSystems_type, categories = known_OperatingSystems)
    OperatingSystems_input = pd.get_dummies(OperatingSystems_type, prefix = 'OperatingSystems', drop_first=True)

    Browser_type = pd.Series([Browser])
    Browser_type = pd.Categorical(Browser_type, categories = known_Browser)
    Browser_input = pd.get_dummies(Browser_type, prefix = 'Browser', drop_first=True)

    Region_type = pd.Series([Region])
    Region_type = pd.Categorical(Region_type, categories = known_Region)
    Region_input = pd.get_dummies(Region_type, prefix = 'Region', drop_first=True)

    VisitorType_type = pd.Series([VisitorType])
    VisitorType_type = pd.Categorical(VisitorType_type, categories = known_VisitorType)
    VisitorType_input = pd.get_dummies(VisitorType_type, prefix = 'VisitorType', drop_first=True)

    Weekend_type = pd.Series([Weekend])
    Weekend_type = pd.Categorical(Weekend_type, categories = known_Weekend)
    Weekend_input = pd.get_dummies(Weekend_type, prefix = 'Weekend', drop_first=True)

    # pre processing cyclical feature
    Month_sin = np.sin(Month*(2.*np.pi/12))
    Month_cos = np.cos(Month*(2.*np.pi/12))

    # concat new data
    onehot_result1 = list(pd.concat([VisitorType_input, Weekend_input], axis = 1).iloc[0])
    onehot_result2 = list(pd.concat([SpecialDay_input, OperatingSystems_input, Browser_input, Region_input], axis = 1).iloc[0])
    new_data = [[Administrative, Administrative_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, TrafficType] + onehot_result1 + [Month_sin, Month_cos] + onehot_result2]

    scaled_input = scaler.transform(new_data)
    prediction = model.predict(scaled_input)

    
    if prediction == 0:
        prediction_not_buying = True
    else:
        prediction_not_buying = False

    output = {0: 'not end up with shopping and will not bring revenue', 1: 'end up shopping and bring revenue'}

    return render_template('form.html', title = 'Prediction', show_prediction = show_prediction, prediction_text = 'The Customer will {}.'.format(output[prediction[0]]), link_active = link_active, prediction_not_buying = prediction_not_buying)

if __name__ == '__main__':
    app.run(debug = True)
