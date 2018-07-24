def NumVarPerf(df,col,target,filepath, truncation=False):
    
    '''
    :param df: the dataset containing numerical independent variable and dependent variable
    :param col: independent variable with numerical type
    :param target: dependent variable, class of 0-1
    :param filepath: the location where we save the histogram
    :param truncation: indication whether we need to do some truncation for outliers
    :return: the descriptive statistics
    '''
    
    # extract target variable and specific indepedent variable
    validDf = df.loc[df[col] == df[col]][[col,target]]
    
    # the percentage of valid elements
    validRcd = validDf.shape[0]*1.0/df.shape[0]
    
    # format the percentage in the form of percent
    validRcdFmt = "%.2f%%"%(validRcd*100)
    
    # the descriptive statistics of each numerical column
    descStats = validDf[col].describe()
    mu = "%.2e" % descStats['mean']
    std = "%.2e" % descStats['std']
    maxVal = "%.2e" % descStats['max']
    minVal = "%.2e" % descStats['min']
    
    # we show the distribution by churn/not churn state
    x = validDf.loc[validDf[target]==1][col]
    y = validDf.loc[validDf[target]==0][col]
    xweights = 100.0 * np.ones_like(x) / x.size
    yweights = 100.0 * np.ones_like(y) / y.size
    
    # if need truncation, truncate the numbers in 95th quantile
    if truncation == True:
        pcnt95 = np.percentile(validDf[col],95)
        x = x.map(lambda x: min(x,pcnt95))
        y = y.map(lambda x: min(x,pcnt95))
        
    fig, ax = pyplot.subplots() 
    ax.hist(x, weights=xweights, alpha=0.5,label='Attrition')
    ax.hist(y, weights=yweights, alpha=0.5,label='Retained')
    titleText = 'Histogram of '+ col +'\n'+'valid pcnt ='+validRcdFmt+', \
             Mean ='+mu + ', Std='+std+'\n max='+maxVal+', min='+minVal
    ax.set(title= titleText, ylabel='% of Dataset in Bin')
    ax.margins(0.05)  
    ax.set_ylim(bottom=0)
    pyplot.legend(loc='upper right')
    figSavePath = filepath+str(col)+'.png'
    pyplot.savefig(figSavePath)
    pyplot.close(1)
 
def CharVarPerf(df,col,target,filepath):
    '''
    :param df: the dataset containing numerical independent variable and dependent variable
    :param col: independent variable with numerical type
    :param target: dependent variable, class of 0-1
    :param filepath: the location where we save the histogram
    :return: the descriptive statistics
    '''

    validDf = df.loc[df[col] == df[col]][[col, target]]
    validRcd = validDf.shape[0]*1.0/df.shape[0]
    recdNum = validDf.shape[0]
    validRcdFmt = "%.2f%%"%(validRcd*100)

    freqDict = {}
    churnRateDict = {}

    #for each category in the categorical variable, we count the percentage and churn rate
    for v in set(validDf[col]):

        vDf = validDf.loc[validDf[col] == v]
        freqDict[v] = vDf.shape[0]*1.0/recdNum
        churnRateDict[v] = sum(vDf[target])*1.0/vDf.shape[0]
    descStats = pd.DataFrame({'percent':freqDict,'churn rate':churnRateDict})
    fig = pyplot.figure()  # Create matplotlib figure
    ax = fig.add_subplot(111)  # Create matplotlib axes
    ax2 = ax.twinx()  # Create another axes that shares the same x-axis as ax.
    pyplot.title('The percentage and churn rate for '+col+'\n valid pcnt ='+validRcdFmt)
    descStats['churn rate'].plot(kind='line', color='red', ax=ax)
    descStats.percent.plot(kind='bar', color='blue', ax=ax2, width=0.2,position = 1)
    ax.set_ylabel('churn rate')
    ax2.set_ylabel('percentage')
    figSavePath = filepath+str(col)+'.png'
    pyplot.savefig(figSavePath)
    pyplot.close(1)

# The function making up missing values in Continuous or Categorical variable
def MakeupMissing(df,col,type,method):
    '''
    :param df: dataset containing columns with missing value
    :param col: columns with missing value
    :param type: the type of the column, should be Continuous or Categorical
    :return: the made up columns
    '''
    # Take the sample with non-missing value in col
    validDf = df.loc[df[col] == df[col]][[col]]
    if validDf.shape[0] == df.shape[0]:
        return 'There is no missing value in {}'.format(col)
    # copy the original value from col to protect the original dataframe
    missingList = [i for i in df[col]]
    if type == 'Continuous':
        if method not in ['Mean','Random']:
            return 'Please specify the correct treatment method for missing continuous variable!'
        # get the descriptive statistics of col
        descStats = validDf[col].describe()
        mu = descStats['mean']
        std = descStats['std']
        maxVal = descStats['max']
        
        # detect the extreme value using 3-sigma method
        if maxVal > mu+3*std:
            for i in list(validDf.index):
                if validDf.loc[i][col] > mu+3*std:
                    #decrease the extreme value to normal level
                    validDf.loc[i][col] = mu + 3 * std
            #re-calculate the mean based on cleaned data
            mu = validDf[col].describe()['mean']
        for i in range(df.shape[0]):
            if df.loc[i][col] != df.loc[i][col]:
                #use the mean or sampled data to replace the missing value
                if method == 'Mean':
                    missingList[i] = mu
                elif method == 'Random':
                    missingList[i] = random.sample(validDf[col],1)[0]
    
    elif type == 'Categorical':
        if method not in ['Mode', 'Random']:
            return 'Please specify the correct treatment method for missing categorical variable!'
        # calculate the probability of each type of the categorical variable
        freqDict = {}
        recdNum = validDf.shape[0]
        for v in set(validDf[col]):
            vDf = validDf.loc[validDf[col] == v]
            freqDict[v] = vDf.shape[0] * 1.0 / recdNum
            
        # find the category with highest probability
        modeVal = max(freqDict.items(), key=lambda x: x[1])[0]
        freqTuple = freqDict.items()
        
        # cumulative sum of each category
        freqList = [0]+[i[1] for i in freqTuple]
        freqCumsum = cumsum(freqList)
        
        for i in range(df.shape[0]):
            if df.loc[i][col] != df.loc[i][col]:
                if method == 'Mode':
                    missingList[i] = modeVal
                if method == 'Random':
                    # determine the sampled category using unifor distributed random variable
                    a = random.random(1)
                    position = [k+1 for k in range(len(freqCumsum)-1) if freqCumsum[k]<a<=freqCumsum[k+1]][0]
                    missingList[i] = freqTuple[position-1][0]
                    
    print 'The missing value in {0} has been made up with the mothod of {1}'.format(col, method)
    return missingList

# Use numerical representative for ategorical variable
def Encoder(df, col, target):
    '''
    :param df: the dataset containing categorical variable
    :param col: the name of categorical variabel
    :param target: class, with value 1 or 0
    :return: the numerical encoding for categorical variable
    '''
    encoder = {}
    for v in set(df[col]):
        if v == v:
            subDf = df[df[col] == v]
        else:
            xList = list(df[col])
            nanInd = [i for i in range(len(xList)) if xList[i] != xList[i]]
            subDf = df.loc[nanInd]
        encoder[v] = sum(subDf[target])*1.0/subDf.shape[0]
    newCol = [encoder[i] for i in df[col]]
    return newCol

# convert the date variable into the days
def Date2Days(df, dateCol, base):
    '''
    :param df: the dataset containing date variable in the format of 2017/1/1
    :param date: the column of date
    :param base: the base date used in calculating day gap
    :return: the days gap
    '''
    base2 = time.strptime(base,'%Y/%m/%d')
    base3 = datetime.datetime(base2[0],base2[1],base2[2])
    date1 = [time.strptime(i,'%Y/%m/%d') for i in df[dateCol]]
    date2 = [datetime.datetime(i[0],i[1],i[2]) for i in date1]
    daysGap = [(date2[i] - base3).days for i in range(len(date2))]
    return daysGap

# Calculate the ratio between two variables
def ColumnDivide(df, colNumerator, colDenominator):
    '''
    :param df: the dataframe containing variable x & y
    :param colNumerator: the numerator variable x
    :param colDenominator: the denominator variable y
    :return: x/y
    '''
    N = df.shape[0]
    rate = [0]*N
    xNum = list(df[colNumerator])
    xDenom = list(df[colDenominator])
    for i in range(N):
        #if the denominator is non-zero, work out the ratio
        if xDenom[i]>0:
            rate[i] = xNum[i]*1.0/xDenom[i]
        # if the denominator is zero, assign 0 to the ratio
        else:
            rate[i] = 0
    return rate