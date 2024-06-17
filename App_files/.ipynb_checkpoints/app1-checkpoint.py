import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from whitenoise import WhiteNoise
import pandas as pd
import os
from support_functions import check_format,investment_options,computer_scatter,computer_pie,user_customization
import dash
from dash import html, dcc
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from werkzeug.urls import url_quote

data_sheet=pd.read_csv(r'rates.csv')
data_values=data_sheet.values
countryList=data_values[:,0]

default_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']
app=dash.Dash(__name__,external_stylesheets=[default_stylesheets,dbc.themes.BOOTSTRAP]) #initialising dash app
server=app.server
server.wsgi_app=WhiteNoise(server.wsgi_app,root=os.path.join(os.getcwd(),'static'),prefix='static/')
app.title='Savings Planner'
app.layout=html.Div(id='parent',children=[
    # html.Img(id='header_icon',className='headerAvatar',src=r'static\app_logo.svg'),
    html.H1('Savings planner', className='headerMain', style={'font-size': '4.5rem', 'line-height': '1.2', 'letter-spacing': '-.1rem', 'margin-bottom': '2rem'}),
    html.Div(id='column_layout',className='row',children=[
        html.Div(id='left_column',className='left',children=[
            html.Div(id='left_column_top',className='block inputBlock',children=[
                html.H3('Input parameters',className='headerBlock headerInput',style={ 'font-size': '3.0rem', 'line-height': '1.3',  'letter-spacing': '-.1rem', 'margin-bottom': '1.5rem', 'margin-top': '1.5rem'}),
                html.Div(id='period_question',children=[
                    html.P('Saving period:',className='textBlock textPeriod',style={'display':'inline-block'}),
                    dcc.Input(id='input_period',className='inputField inputPeriod',type='text',placeholder=' e.g., 10 years, 15',style={'display':'inline-block'})
                    ]),
                html.Div(id='goal_question',children=[
                    html.P('Saving goal:',className='textBlock textSaving',style={'display':'inline-block'}),
                    dcc.Input(id='input_goal',className='inputField inputGoal',type='text',placeholder=' e.g., 10000, 1000000',style={'display':'inline-block'})
                    ]),
                html.Div(id='start_question',children=[
                    html.P('Starting amount:',className='textBlock textStarting',style={'display':'inline-block'}),
                    dcc.Input(id='input_start',className='inputField inputStart',type='text',placeholder=' e.g., 0, 200000',style={'display':'inline-block'})
                    ]),
                dbc.Button('Submit request', id='submit_button', className='buttonSubmit', n_clicks=0, 
                    style={
                    'display': 'inline-block',
                    'height': '38px',
                    'padding': '0 30px',
                    'color': '#555',
                    'text-align': 'center',
                    'font-size': '11px',
                    'font-weight': '600',
                    'line-height': '38px',
                    'letter-spacing': '.1rem',
                    'text-transform': 'uppercase',
                    'text-decoration': 'none',
                    'white-space': 'nowrap',
                    'background-color': 'transparent',
                    'border-radius': '4px',
                    'border': '1px solid #bbb',
                    'cursor': 'pointer',
                    'box-sizing': 'border-box'
                }),
                dbc.Button('Reset',id='reset_button',className='buttonReset',n_clicks=0,style={'display':'inline-block'},)
                ]),
            dbc.Collapse(id='hide_left_choice3',children=[
                html.Div(id='left_column_choice3',className='block parameterBlock3',children=[
                    html.H3('Parameters',className='headerBlock headerParameterBlock3',style={ 'font-size': '3.0rem', 'line-height': '1.3',  'letter-spacing': '-.1rem', 'margin-bottom': '1.5rem', 'margin-top': '1.5rem'}),
                    html.Div(id='bank_bill',className='bankBill',children=[
                        html.Div(id='bank_div',className='bankDiv',children=[
                            html.H5('Bank',className='subHeaderBlock subHeaderBank',style={'font-size': '2.2rem', 'line-height': '1.5', 'letter-spacing': '-.05rem', 'margin-bottom': '0.6rem', 'margin-top': '0.6rem'}),
                            html.H6('Share',className='subsubHeaderBlock subsubHeaderBank'),
                            dcc.Slider(id='share_bank',className='bankSlider',min=0,max=100,step=1,marks=None,
                                       tooltip={'placement':'bottom','always_visible':True})
                            ],style={'display':'inline-block'}),
                        html.Div(id='bill_div',className='billDiv',children=[
                            html.H5('Treasury bills',className='subHeaderBlock subHeaderBill',style={'font-size': '2.2rem', 'line-height': '1.5', 'letter-spacing': '-.05rem', 'margin-bottom': '0.6rem', 'margin-top': '0.6rem'}),
                            html.H6('Share',className='subsubHeaderBlock subsubHeaderBill',style={'font-size': '2.0rem', 'line-height': '1.6', 'letter-spacing': '0', 'margin-bottom': '0.75rem', 'margin-top': '0.75rem'}),
                            dcc.Slider(id='share_bill',className='billSlider',min=0,max=100,step=1,marks=None,
                                       tooltip={'placement':'bottom','always_visible':True})
                            ],style={'display':'inline-block'})
                        ]),
                    html.H5('Corporate bonds',className='subHeaderBlock subHeaderBond',style={'font-size': '2.2rem', 'line-height': '1.5', 'letter-spacing': '-.05rem', 'margin-bottom': '0.6rem', 'margin-top': '0.6rem'}),
                    html.Div(id='bond',className='bondSection',children=[
                        html.Div(id='bond_share_div',className='bondShareDiv',children=[
                            html.H6('Share',className='subsubHeaderBlock subsubHeaderBondShare',style={'font-size': '2.0rem', 'line-height': '1.6', 'letter-spacing': '0', 'margin-bottom': '0.75rem', 'margin-top': '0.75rem'}),
                            dcc.Slider(id='share_bond',className='bondSliderShare',min=0,max=100,step=1,marks=None,
                                       tooltip={'placement':'bottom','always_visible':True}),
                            ],style={'display':'inline-block'}),
                        html.Div(id='bond_rate_div',className='bondRateDiv',children=[
                            html.H6('Rate of return',className='subsubHeaderBlock subsubHeaderBondRates',style={'font-size': '2.0rem', 'line-height': '1.6', 'letter-spacing': '0', 'margin-bottom': '0.75rem', 'margin-top': '0.75rem'}),
                            dcc.Slider(id='rate_bond',className='bondSliderRates',min=0,max=50,step=0.1,marks=None,
                                       tooltip={'placement':'bottom','always_visible':True}),
                            ],style={'display':'inline-block'})
                        ]),
                    html.H5('Real estate',className='subHeaderBlock subHeaderEstate',style={'font-size': '2.2rem', 'line-height': '1.5', 'letter-spacing': '-.05rem', 'margin-bottom': '0.6rem', 'margin-top': '0.6rem'}),
                    html.Div(id='estate',className='estateSection',children=[
                        html.Div(id='estate_share_div',className='estateShareDiv',children=[
                            html.H6('Share',className='subsubHeaderBlock subsubHeaderEstateShare',style={'font-size': '2.0rem', 'line-height': '1.6', 'letter-spacing': '0', 'margin-bottom': '0.75rem', 'margin-top': '0.75rem'}),
                            dcc.Slider(id='share_estate',className='estateSliderShare',min=0,max=100,step=1,marks=None,
                                       tooltip={'placement':'bottom','always_visible':True}),
                            ],style={'display':'inline-block'}),
                        html.Div(id='estate_rate_div',className='estateRateDiv',children=[
                            html.H6('Rate of return',className='subsubHeaderBlock subsubHeaderEstateRates',style={'font-size': '2.0rem', 'line-height': '1.6', 'letter-spacing': '0', 'margin-bottom': '0.75rem', 'margin-top': '0.75rem'}),
                            dcc.Slider(id='rate_estate',className='estateSliderRates',min=0,max=50,step=0.1,marks=None,
                                       tooltip={'placement':'bottom','always_visible':True}),
                            ],style={'display':'inline-block'})
                        ]),
                    html.H5('SPY',className='subHeaderBlock subHeaderSpy',style={'font-size': '2.2rem', 'line-height': '1.5', 'letter-spacing': '-.05rem', 'margin-bottom': '0.6rem', 'margin-top': '0.6rem'}),
                    html.Div(id='spy',className='spySection',children=[
                        html.Div(id='spy_share_div',className='spyShareDiv',children=[
                            html.H6('Share',className='subsubHeaderBlock subsubHeaderSpyShare',style={'font-size': '2.0rem', 'line-height': '1.6', 'letter-spacing': '0', 'margin-bottom': '0.75rem', 'margin-top': '0.75rem'}),
                            dcc.Slider(id='share_spy',className='spySliderShare',min=0,max=100,step=1,marks=None,
                                       tooltip={'placement':'bottom','always_visible':True}),
                            ],style={'display':'inline-block'}),
                        html.Div(id='spy_rate_div',className='spyRateDiv',children=[
                            html.H6('Rate of return',className='subsubHeaderBlock subsubHeaderSpyRates',style={'font-size': '2.0rem', 'line-height': '1.6', 'letter-spacing': '0', 'margin-bottom': '0.75rem', 'margin-top': '0.75rem'}),
                            dcc.Slider(id='rate_spy',className='spySliderRates',min=0,max=50,step=0.1,marks=None,
                                       tooltip={'placement':'bottom','always_visible':True}),
                            ],style={'display':'inline-block'})
                        ]),
                    html.H5('BTC', className='subHeaderBlock subHeaderBTC', style={'font-size': '2.2rem', 'line-height': '1.5', 'letter-spacing': '-.05rem', 'margin-bottom': '0.6rem', 'margin-top': '0.6rem'}),
                    html.Div(id='BTC',className='BTCSection',children=[
                        html.Div(id='BTC_share_div',className='BTCdShareDiv',children=[
                            html.H6('Share',className='subsubHeaderBlock subsubHeaderBTCShare',style={'font-size': '2.0rem', 'line-height': '1.6', 'letter-spacing': '0', 'margin-bottom': '0.75rem', 'margin-top': '0.75rem'}),
                            dcc.Slider(id='share_BTC',className='BTCSliderShare',min=0,max=100,step=1,marks=None,
                                       tooltip={'placement':'bottom','always_visible':True}),
                            ],style={'display':'inline-block'}),
                        html.Div(id='BTC_rate_div',className='BTCRateDiv',children=[
                            html.H6('Rate of return',className='subsubHeaderBlock subsubHeaderBTCRates',style={'font-size': '2.0rem', 'line-height': '1.6', 'letter-spacing': '0', 'margin-bottom': '0.75rem', 'margin-top': '0.75rem'}),
                            dcc.Slider(id='rate_BTC',className='BTCSliderRates',min=0,max=50,step=0.1,marks=None,
                                       tooltip={'placement':'bottom','always_visible':True}),
                            ],style={'display':'inline-block'})
                        ])
                    ])
                ],is_open=False)
            ]),
        html.Div(id='right_column',className='right',children=[
            dbc.Collapse(id='hide_right',children=[
                dbc.Collapse(id='hide_choice',children=[
                    html.Div(id='choice_1',className='block choice choice1',children=[
                        html.H3('Investment\noptions',className='headerBlock headerChoice1',style={ 'font-size': '3.0rem', 'line-height': '1.3',  'letter-spacing': '-.1rem', 'margin-bottom': '1.5rem', 'margin-top': '1.5rem'}),
                        # html.Img(id='choice1_icon',className='choiceLogo',src=r'static\choice1_logo.svg'),
                        html.Br(),
                        dbc.Button('Click here',id='choice1_button',className='buttonChoice button1',n_clicks=0)
                        ],style={'display':'inline-block'}),
                    html.Div(id='choice_2',className='block choice choice2',children=[
                        html.H3('Computer\'s\nsuggestion',className='headerBlock headerChoice2',style={ 'font-size': '3.0rem', 'line-height': '1.3',  'letter-spacing': '-.1rem', 'margin-bottom': '1.5rem', 'margin-top': '1.5rem'}),
                        # html.Img(id='choice2_icon',className='choiceLogo',src=r'static\choice2_logo.svg'),
                        html.Br(),
                        dbc.Button('Click here',id='choice2_button',className='buttonChoice button2',n_clicks=0)
                        ],style={'display':'inline-block'}),
                    html.Div(id='choice_3',className='block choice choice3',children=[
                        html.H3('User\'s\ncustomization',className='headerBlock headerChoice3',style={ 'font-size': '3.0rem', 'line-height': '1.3',  'letter-spacing': '-.1rem', 'margin-bottom': '1.5rem', 'margin-top': '1.5rem'}),
                        # html.Img(id='choice3_icon',className='choiceLogo',src=r'static\choice3_logo.svg'),
                        html.Br(),
                        dbc.Button('Click here',id='choice3_button',className='buttonChoice button3',n_clicks=0)
                        ],style={'display':'inline-block'})
                    ],is_open=True),
                dbc.Collapse(id='hide_choice1',children=[
                    html.Div(id='graph1_div',className='block graphChoice1',children=[
                        dcc.Graph(id='graph1',className='graphChoice1Scatter',figure={})
                        ])
                    ],is_open=False),
                dbc.Collapse(id='hide_choice2',children=[
                    html.Div(id='graph2_div',className='block graphChoice2',children=[
                        dcc.Graph(id='graph2_scatter',className='graphChoice2Scatter',figure={},style={'display':'inline-block'}),
                        dcc.Graph(id='graph2_pie',className='graphChoice2Pie',figure={},style={'display':'inline-block'})
                        ])
                    ],is_open=False),
                dbc.Collapse(id='hide_choice3',children=[
                    html.Div(id='graph3_div',className='block graphChoice3',children=[
                        dcc.Graph(id='graph3',className='graphChoice3Scatter',figure={})
                        ])
                    ],is_open=False),
                dbc.Collapse(id='hide_return',children=[
                    dbc.Button('Return to the choices',id='back_choice',className='buttonPrevious',n_clicks=0)
                    ],is_open=False)
                ],is_open=False),
            dbc.Collapse(id='hide_error_input',children=[
                html.Div(id='middle_column_input',className='block errorBlock',children=[
                    html.H3('Error',id='error_heading_input',className='headerError',style={ 'font-size': '3.0rem', 'line-height': '1.3',  'letter-spacing': '-.1rem', 'margin-bottom': '1.5rem', 'margin-top': '1.5rem'}),
                    html.Img(id='error_icon_input',className='errorLogo',src=r'static\error_logo.svg'),
                    html.P(id='error_text_input',className='errorMessage'),
                    dbc.Collapse(id='hide_contact_link',children=[
                        # html.A('Contact me.',href='https://etienneauroux.com/#contactme',id='contact_country',className='linkContact',target='_blank')
                        ],is_open=False)
                    ]),
                html.Div(id='mask_input',className='maskDiv')
                ],is_open=False),
            dbc.Collapse(id='hide_error_repartition',children=[
                html.Div(id='middle_column_repartition',className='block errorBlock',children=[
                    html.H3('Error',id='error_heading_repartition',className='headerError',style={ 'font-size': '3.0rem', 'line-height': '1.3',  'letter-spacing': '-.1rem', 'margin-bottom': '1.5rem', 'margin-top': '1.5rem'}),
                    html.Img(id='error_icon_repartition',className='errorLogo',src=r'static\error_logo.svg'),
                    html.P('Your portfolio is bigger than 100 %, please adjust your shares accordingly.',id='error_text_repartition',className='errorMessage')
                    ]),
                html.Div(id='mask_repartition',className='maskDiv')
                ],is_open=False)            
            ])
        ]),
    
    ])
@app.callback(
    Output('submit_button','n_clicks'),
    Input('reset_button','n_clicks')
)
def reset_layout(clicks):
    if clicks==0:
        raise PreventUpdate
    else:
        return 0
@app.callback(
    Output('submit_button','disabled'),
    Input('submit_button','n_clicks')
)
def submit_switch(clicks):
    if clicks==0:
        switch=False
    else:
        switch=True
    return switch
@app.callback(
    [Output('hide_error_input','is_open'),
     Output('error_text_input','children'),
     Output('hide_right','is_open'),
     Output('hide_contact_link','is_open')
    ],
    Input('submit_button','n_clicks'),
    [State('input_period','value'),
     State('input_goal','value'),
     State('input_start','value')]
)
def toggle_choice(clicks,time_goal,money_goal,money_start):
    if clicks==0:
        switch_error=False
        message_error=''
        switch_choice=False
        switch_link=False
    else:
        switch_error=True
        switch_choice=False
        switch_link=False
        test_result=check_format(money_start,money_goal,time_goal)
        if test_result=='format_money_start':
            message_error='The starting amount must be a number.'
        elif test_result=='format_money_goal':
            message_error='The saving goal must be a number.'
        elif test_result=='format_money_both':
            message_error='The starting amount and the saving goal must be numbers.'
        elif test_result=='money_goal<=0':
            message_error='The saving goal must be bigger than zero.'
        elif test_result=='money_start<0':
            message_error='The starting amount must be bigger than zero.'
        elif test_result=='money_both<=0':
            message_error='The saving goal and starting amount must be bigger than zero.'
        elif test_result=='goal<=start':
            message_error='The saving goal must be bigger than the starting amount.'
        elif test_result=='format_period':
            message_error='The saving period must be a number of years, for example: \"10 years\".'
        elif test_result=='amount_period':
            message_error='Wrong format for the starting amount and/or (saving goal) and the saving period.'
            switch_link=True
        else:
            message_error=''
            switch_error=False
            switch_choice=True
    return switch_error,message_error,switch_choice,switch_link

@app.callback(
    Output('hide_return','is_open'),
    [Input('choice1_button','n_clicks'),
     Input('choice2_button','n_clicks'),
     Input('choice3_button','n_clicks')]
)
def toggle_return_button(clicks1,clicks2,clicks3):
    if clicks1==0 and clicks2==0 and clicks3==0:
        return False
    else:
        return True

@app.callback( #User has chosen "Investment options"
    [Output('hide_choice1','is_open'),
     Output('graph1','figure')],
    [Input('choice1_button','n_clicks'),
     Input('submit_button','n_clicks')],
    [State('input_period','value'),
     State('input_goal','value'),
     State('input_start','value')]
)
def toggle_choice1(clicks1,submit,time_goal,money_goal,money_start):
    if clicks1==0:
        scatter={}
        switch_graph=False
    elif submit==0:
        scatter={}
        switch_graph=False
    else:
        scatter=investment_options(money_start,money_goal,time_goal)
        switch_graph=True
    return switch_graph,go.Figure(data=scatter)

@app.callback( #User has chosen "Computer suggestion"
    [Output('hide_choice2','is_open'),
     Output('graph2_scatter','figure'),
     Output('graph2_pie','figure')],
    [Input('choice2_button','n_clicks'),
     Input('submit_button','n_clicks'),
     Input('graph2_scatter','hoverData')],
    [State('input_period','value'),
     State('input_goal','value'),
     State('input_start','value')]
)
def toggle_choice2(clicks2,submit,hoverData,time_goal,money_goal,money_start):
    if clicks2==0:
        switch_graph=False
        # switch_parameter=False
        scatter={}
        pie={}
    elif submit==0:
        switch_graph=False
        # switch_parameter=False
        scatter={}
        pie={}
    else:
        switch_graph=True
        switch_parameter=True
        
        if hoverData is None:
            pie=computer_pie(money_start,money_goal,time_goal,0)
        else:
            hover_data = hoverData["points"][0]
            num=hover_data["pointNumber"]
            pie=computer_pie(money_start,money_goal,time_goal,int(num))
        scatter,message=computer_scatter(money_start,money_goal,time_goal)
    return switch_graph,go.Figure(data=scatter),go.Figure(data=pie)

@app.callback(
    [Output('share_bank','value'),
     Output('share_bill','value'),
     Output('share_bond','value'),
     Output('share_estate','value'),
     Output('share_spy','value'),
     Output('share_BTC','value'),
     Output('rate_bond','value'),
     Output('rate_estate','value'),
     Output('rate_spy','value'),
     Output('rate_BTC','value')],
    Input('submit_button','n_clicks')
)
def default_slider_values(clicks):
    data_sheet=pd.read_csv('rates.csv')
    data_values= data_sheet.values

    if clicks==0:
        raise PreventUpdate
    else:
        choice= 'Canada'
        sbank=0
        sbill=30
        sbond=30
        sestate=20
        sspy=20
        sbtc = 20
        rbond= data_values[:,3][0]
        restate= data_values[:,4][0]
        rspy= data_values[:,5][0]
        rbtc = data_values[:,6][0]
        return sbank, sbill, sbond, sestate, sspy, sbtc, rbond, restate, rspy, rbtc

@app.callback( #User has chosen "User customization"
    [Output('hide_choice3','is_open'),
      Output('hide_left_choice3','is_open'),
      Output('graph3','figure'),
      Output('hide_error_repartition','is_open')],
    [Input('choice3_button','n_clicks'),
     Input('submit_button','n_clicks'),
      Input('share_bank','value'),
      Input('share_bill','value'),
      Input('share_bond','value'),
      Input('share_estate','value'),
      Input('share_spy','value'),
      Input('share_BTC','value'),
      Input('rate_bond','value'),
      Input('rate_estate','value'),
      Input('rate_spy','value'),
      Input('rate_BTC','value')],
    [State('input_period','value'),
      State('input_goal','value'),
      State('input_start','value')]
)
def toggle_choice3(clicks3,submit,sbank,sbill,sbond,sestate,sspy,sbtc,rbond,restate,rspy,rbtc,time_goal,money_goal,money_start):
    if clicks3==0:
        switch_graph=False
        switch_parameter=False
        switch_error=False
        scatter={}
    elif submit==0:
        switch_graph=False
        switch_parameter=False
        switch_error=False
        scatter={}
    else:
        switch_parameter=True
        repartition=[sbill,sbond,sestate,sspy]
        if sum(repartition)>100:
            switch_graph=False
            switch_error=True
            scatter={}
        else:
            switch_graph=True
            switch_error=False
            rates=[rbond,restate,rspy]
            bank=sbank
            scatter=user_customization(money_start,money_goal,time_goal,rates,bank,repartition)
    return switch_graph,switch_parameter,go.Figure(data=scatter),switch_error

@app.callback( #User wants to go back to choices
    [Output('choice1_button','n_clicks'),
     Output('choice2_button','n_clicks'),
     Output('choice3_button','n_clicks')],
    Input('back_choice','n_clicks')
)
def return_to_choices(clicks):
    if clicks:
        clicks1=0
        clicks2=0
        clicks3=0
        return clicks1,clicks2,clicks3
    else:
        raise PreventUpdate

port=8049
if __name__ =='__main__':
    app.run_server(debug=True,port=port,use_reloader=False)