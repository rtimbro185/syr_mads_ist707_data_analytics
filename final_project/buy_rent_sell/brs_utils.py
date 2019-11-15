# -*- coding: utf-8 -*-
# RTIMBROO UTIL FUNCTIONS

#----------------------------------------------------#
#### TIME SERIES FUNCTIONS USING FACEBOOK PROPHET ####
#----------------------------------------------------#

'''
# Facebook Prophet requires columns to be in a specific format
The input to Prophet is always a dataframe with two columns: ds and y. The ds (datestamp) column should be of a format expected by Pandas, 
ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. 
The y column must be numeric, and represents the measurement we wish to forecast.
'''
def dfTransformForProphet(df,cols,index,logger):
    import pandas as pd
    try:
        df = df.drop(columns=cols)
    except KeyError:
        pass
    df = df.set_index(index)
    df = df.T
    df.index = pd.to_datetime(df.index)
    return (df)
#------------------------------------------------

'''
# util function for getting the Date series values
'''

# util function for getting the Date series values
def getDateColumns(series,d):   
    return([i for i in series if d in i])
#---------------------------------------------

'''
# function to format and create a prophet model
'''
# function to format and create a prophet model
def beProphet(label,components,modelOutDir,logger, predPeriods=12, log_of_value=True):
    import pickle
    import pandas as pd
    import numpy as np
    from fbprophet import Prophet
    
    model={}
    # restructure the dataframe to fit prophet
    df = pd.DataFrame(components)
    df = df.reset_index()
    
    # condition on taking the log or not
    if log_of_value:
        value = 'log_'+label
        df[value] = np.log(df[label])
    dfProphet = df.rename(index=str, columns={value:'y','index':'ds'})
    dfProphet = dfProphet.loc[:,['y','ds']]
    
    # get prices annual delta
    
    
    # setting uncertainty interval to 95%
    zipModel = Prophet(interval_width=0.95)
    zipModel_fit = zipModel.fit(dfProphet)
    model['model_fit'] = zipModel_fit
    # make future dates dataframe
    future_dates = zipModel.make_future_dataframe(periods=predPeriods, freq='M', include_history=True)
    
    # model
    forecast = zipModel.predict(future_dates)
    model['model_forecast'] = forecast
    
    # predicted versus actual
    
    
    #save model to file
    with open(modelOutDir+'_fit','wb') as f:
        pickle.dump(zipModel_fit,f)
    
    with open(modelOutDir+'_forecast','wb') as f:
        pickle.dump(forecast,f)
    
    return(model)
#---------------------------------------------


'''
Calculate annual price value change by zipcode
'''
# Calculate the yearly price changes
def calcPriceDelta(df,dateSeries, logger, year_start=1997, year_end=2020):
    import pandas as pd
    
    years = [str(i) for i in range(year_start,year_end)]
    dateCols = {}
    for y in years:
        dateCols[y] = getDateColumns(dateSeries,y)
    
    yearAvg = {}
    for d in dateCols:
        subSet = df[dateSeries.isin(dateCols[d])]
        anualAvg = subSet.iloc[:,1].mean()
        yearAvg[d] = anualAvg
        #break
    #print('Yearly Price Averages: {0}\n'.format(yearAvg))

    thisYear = ''
    priorYear = ''
    prior_years = []
    priceDelta = {}
    priceDeltaPercent = {}
    for i, year in enumerate(yearAvg):
        thisYear = year
        
        if not i == 0:
            priorYear = prior_years[i-1] # not first year
        else:
            priorYear = year # is first year
        #print('This Year: {0}'.format(thisYear))
        #print('Prior Year: {0}\n'.format(priorYear))
        
        # set prior year list
        prior_years.append(thisYear)
    
        # calculate delta between years
        #print('This Year Average Price: {0}'.format(yearAvg[thisYear]))
        #print('Prior Year Average Price: {0}\n'.format(yearAvg[priorYear]))
        
        delta = yearAvg[thisYear] - yearAvg[priorYear]
        percentDelta = delta/yearAvg[thisYear]*100
        
        #print('This Year: {0}, Prior Year: {1}, Price Delta: {2}\n'.format(thisYear,priorYear,delta))
        
        priceDelta[thisYear] = delta
        priceDeltaPercent[thisYear] = percentDelta
    #print('Yearly Price Change:{0}\n'.format(priceDelta))
    return(pd.DataFrame([yearAvg,priceDelta,priceDeltaPercent], index=['Yearly_Price_Avg','Yearly_Price_Delta',
                                                                       'Yearly_Price_Delta_Percent']))

#---------------------------------------------


# Calculate prior five year net value changes
def calcPriorYearNetChange(label,components,dataOutDir,logger,year_start=2014,year_end=2019):
    import pandas as pd
    
    years = [str(i) for i in range(year_start,year_end)]
    #print(years)
    dateCols = {}
    
    # restructure the dataframe
    df = pd.DataFrame(components)
    df = df.reset_index()
    
    prices = df.rename(index=str, columns={label:'Median_Price','index':'Date'})
    prices = prices.loc[:,['Date','Median_Price']]
    prices['Date'] = prices.Date.astype(str)
    
    #filter the Date series to the last five years
    for y in years:
        dateCols[y] = getDateColumns(prices.Date,y)
    
    # get the yearly avg subset
    yearAvg = {}
    for d in dateCols:
        subSet = prices[prices.Date.isin(dateCols[d])]
        anualAvg = subSet.iloc[:,1].mean()
        yearAvg[d] = anualAvg
        #break
    #print('Yearly Price Averages: {0}\n'.format(yearAvg))
    
    deltas = calcYearlyDelta(yearAvg)
    #print(deltas[1])
    
    # deltas[0] is the price delta, deltas[1] is the the percent delta
    priorAvgPercentDeltas = {}
    priorYearPercentAvg = 0.0
    prior2YearPercentAvg = 0.0
    prior3YearPercentAvg = 0.0
    prior4YearPercentAvg = 0.0
    prior5YearPercentAvg = 0.0
    sortedDeltaKeys = reversed(sorted(deltas[1].keys()))
    for i, key in enumerate(sortedDeltaKeys):
        if i == 0: 
            priorYearPercentAvg = deltas[1][key];
            priorAvgPercentDeltas['OneYearAvg'] = priorYearPercentAvg
        if i == 1: 
            #print(deltas[1][key])
            #print(np.mean([priorYearPercentAvg,deltas[1][key]]))
            prior2YearPercentAvg = np.mean([priorYearPercentAvg,deltas[1][key]])
            priorAvgPercentDeltas['TwoYearAvg'] = prior2YearPercentAvg
        if i == 2: 
            #print(deltas[1][key])
            #print(np.mean([priorYearPercentAvg,prior2YearPercentAvg,deltas[1][key]]))
            prior3YearPercentAvg = np.mean([priorYearPercentAvg,prior2YearPercentAvg,deltas[1][key]])
            priorAvgPercentDeltas['ThreeYearAvg'] = prior3YearPercentAvg
        if i == 3: 
            #print(deltas[1][key])
            #print(np.mean([priorYearPercentAvg,prior2YearPercentAvg,prior3YearPercentAvg,deltas[1][key]]))
            prior4YearPercentAvg = np.mean([priorYearPercentAvg,prior2YearPercentAvg,prior3YearPercentAvg,deltas[1][key]])
            priorAvgPercentDeltas['FourYearAvg'] = prior4YearPercentAvg
        if i == 4: 
            #print(deltas[1][key])
            #print(np.mean([priorYearPercentAvg,prior2YearPercentAvg,prior3YearPercentAvg,prior4YearPercentAvg,deltas[1][key]]))
            prior5YearPercentAvg = np.mean([priorYearPercentAvg,prior2YearPercentAvg,prior3YearPercentAvg,prior4YearPercentAvg,deltas[1][key]])
            priorAvgPercentDeltas['FiveYearAvg'] = prior5YearPercentAvg
     
    return(priorAvgPercentDeltas)
#--------------------------------------------


# Calculate yearly deltas
def calcYearlyDelta(yearlyAvgPrices, logger):
    thisYear = ''
    priorYear = ''
    prior_years = []
    priceDelta = {}
    priceDeltaPercent = {}
    for i, year in enumerate(yearlyAvgPrices):
        thisYear = year
        
        if not i == 0:
            priorYear = prior_years[i-1] # not first year
        else:
            priorYear = year # is first year
        #print('This Year: {0}'.format(thisYear))
        #print('Prior Year: {0}\n'.format(priorYear))
        
        # set prior year list
        prior_years.append(thisYear)
    
        # calculate delta between years
        #print('This Year Average Price: {0}'.format(yearAvg[thisYear]))
        #print('Prior Year Average Price: {0}\n'.format(yearAvg[priorYear]))
        
        delta = yearlyAvgPrices[thisYear] - yearlyAvgPrices[priorYear]
        percentDelta = delta/yearlyAvgPrices[thisYear]*100
        
        #print('This Year: {0}, Prior Year: {1}, Price Delta: {2}\n'.format(thisYear,priorYear,delta))
        
        priceDelta[thisYear] = delta
        priceDeltaPercent[thisYear] = percentDelta
    #print('Yearly Price Change:{0}\n'.format(priceDelta))
    return(priceDelta,priceDeltaPercent)

def calcAnnualPriceAvgChange(regionDf, logger):
    # calculate annual price average and delta change from prior year
    regionPriceDeltas = {}
    ypd1 = pd.DataFrame()
    for i, metro in enumerate(regionDf.columns):
        m_i = regionDf.loc[:,metro]
        m_i = m_i.reset_index()
        m_i = m_i.rename(index=str,columns={'index':'Date'})
        m_i.Date = m_i.Date.astype(str)
        ypd = calcPriceDelta(m_i,m_i.Date)
        ypd = ypd.T
        ypd = ypd.reset_index()
        ypd = ypd.rename(index=str,columns={'index':'Date'})
        ypd = ypd.rename(index=str,columns={'Yearly_Price_Avg':metro+'_Avg',
                                            'Yearly_Price_Delta':metro+'_Delta',
                                           'Yearly_Price_Delta_Percent':metro+'_Delta_Percent'})
        #print(ypd.head())   
        
        ypd1['Date'] = ypd['Date']
        ypd1[metro+'_Avg'] = ypd[metro+'_Avg']
        ypd1[metro+'_Delta'] = ypd[metro+'_Delta']
        ypd1[metro+'_Delta_Percent'] = ypd[metro+'_Delta_Percent']   
        
    return ypd1   

'''

'''
#imageDir
def plotFit(zipcode, fit,forecast,title,imageDir):
    import matplotlib.pyplot as plt
     
    plt.figure(figsize=(20,15))
    p_fit = fit.plot(forecast,uncertainty=True)
    ax = p_fit.get_axes()
    ax[0].set_title(title, fontsize="15", color="black", horizontalalignment='center', verticalalignment='top')
    ax[0].set_xlabel('ZipCode: '+zipcode+' | '+'Date')
    ax[0].set_ylabel('Log Mean Home Prices')
   
    plt.savefig(imageDir+title+'_fit_plot.png')
    plt.show()
    
    
    pc_fit = fit.plot_components(forecast)
    ax = pc_fit.get_axes()
    ax[0].set_title(title)
    ax[0].set_xlabel('ZipCode: '+zipcode+' | '+'Date')
    #ax[0].set_ylabel('Log Mean Home Prices')
    #plt.figure(figsize=(16,6))
    plt.savefig(imageDir+title+'_fit_component_plot.png')
    plt.show()
   
