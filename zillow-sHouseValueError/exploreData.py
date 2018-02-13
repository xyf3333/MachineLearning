import pandas as pd
import os
from plot import plotScatter
from plot import plotBarChart

class exploreData (object):
    """
    """
    def __init__(self, path, featurefile, labelfile2016, labelfile2017):
        self.path = path

        self.featurefile = featurefile
        featuresfile = os.path.abspath(os.path.join(self.path, self.featurefile))
        self.featuresdata = pd.read_csv(featuresfile, low_memory=False)

        self.labelfile2016 = labelfile2016
        labelfile2016 = os.path.abspath(os.path.join(self.path, self.labelfile2016))
        self.labeldata2016 = pd.read_csv(labelfile2016, low_memory=False)

        self.labelfile2017 = labelfile2017
        labelfile2017 = os.path.abspath(os.path.join(self.path, self.labelfile2017))
        self.labeldata2017 = pd.read_csv(labelfile2017, low_memory=False)


    def getFeatureInfo(self):

        # Get list of features
        self.propertiesList = self.featuresdata.columns.values
        # print "The list of feature",  self.propertiesList

        # Total number of  features
        propertiesLen = len(self.propertiesList)
        print 'Total number of features:', propertiesLen

        # Total number of properties
        n_records = len(self.featuresdata.index)
        print "Total number of properties", n_records

        list =[ 'airconditioningtypeid', 'architecturalstyletypeid', 'basementsqft', 'bathroomcnt', 'bedroomcnt', 'buildingclasstypeid', 'buildingqualitytypeid', 'calculatedbathnbr', 'decktypeid', 'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet', 'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15', 'finishedsquarefeet50', 'finishedsquarefeet6', 'fips', 'fireplacecnt', 'fullbathcnt', 'garagecarcnt', 'garagetotalsqft', 'hashottuborspa', 'heatingorsystemtypeid', 'latitude', 'longitude', 'lotsizesquarefeet', 'poolcnt', 'poolsizesum', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'propertycountylandusecode', 'propertylandusetypeid', 'propertyzoningdesc', 'rawcensustractandblock', 'regionidcity', 'regionidcounty', 'regionidneighborhood', 'regionidzip', 'roomcnt', 'storytypeid', 'threequarterbathnbr', 'typeconstructiontypeid', 'unitcnt', 'yardbuildingsqft17', 'yardbuildingsqft26', 'yearbuilt', 'numberofstories', 'fireplaceflag', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'assessmentyear', 'landtaxvaluedollarcnt', 'taxamount', 'taxdelinquencyflag', 'taxdelinquencyyear', 'censustractandblock']
        self.featuresdata = pd.DataFrame.from_records(self.featuresdata, index='parcelid')

        #Get number of every property
        featurelist = []
        self.countProperties = dict()
        print 'Amount of feature value > 2500000'
        print '===Feature,  Amount ==='
        for feature in list:
            self.countProperties[feature] =  self.featuresdata[feature].notnull().sum()
            if self.countProperties[feature] > 2500000:
                featurelist.append(feature)
                print feature, self.countProperties[feature]
        featurelist.append('parcelid')

        #show number of every feature
        pl_1 = plotBarChart(self.countProperties)
        pl_1.get()
        return featurelist

    def getLabelsInfo(self):

        # plot scatter
        plt2 = plotScatter(self.labeldata2016.index, self.labeldata2016['logerror'])
        plt2.get()

        plt3 = plotScatter(self.labeldata2017.index, self.labeldata2017['logerror'])
        plt3.get()

        self.labeldata2016 = pd.DataFrame.from_records(self.labeldata2016, index='parcelid')

        #The list of colume in 2016
        self.labeldata2016List = self.labeldata2016.columns.values

        # Amount of transactions in 2016
        self.labelAmount2016 = len(self.labeldata2016)
        print "Amount of transactions in 2016:", self.labelAmount2016


        #check NaN value
        ref = self.labelAmount2016
        for feature in self.labeldata2016List:
            temp = self.labeldata2016[feature].notnull().sum()
            if ref != temp:
                print'Exist NaN values'


        self.labeldata2017 = pd.DataFrame.from_records(self.labeldata2017, index='parcelid')

        # The list of colume in2017
        self.labeldata2017List = self.labeldata2017.columns.values

        # Amount of transactions in 2017
        self.labelAmount2017 = len(self.labeldata2017)
        print "Amount of transactions in 2017:", self.labelAmount2017

        # check NaN value
        ref = self.labelAmount2017
        for colume in self.labeldata2017List:
            temp = self.labeldata2017[colume].notnull().sum()
            if ref != temp:
                print'Exist NaN values'