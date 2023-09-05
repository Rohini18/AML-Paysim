import pandas as pd 
from sklearn.preprocessing import LabelEncoder

class AMI:
    def __init__(self):
        self.StringColumns = []
        self.readFilePath = ""
        self.writeFilePath = ""
        self.dataframe = ""
        
    def read_dataframe(self):
        print("Reading Dataframe from : ",self.readFilePath)
        self.dataframe = pd.read_csv(self.readFilePath)
        return self.dataframe
    
    def analysis_dataframe(self):
        print("\n Analysing Dataframe")
        print("\n Dataframe")
        print(self.dataframe)
        print("\n Dataframe Shape : \n", self.dataframe.shape)
        print("\n Dataframe Data Types : \n",self.dataframe.dtypes)
        print("\n Dataframe Duplicate Records : \n",self.dataframe.duplicated().sum())
        print("\n Dataframe Null Records : \n",self.dataframe.isnull().values.any())

    def transform_dataframe_str2num(self):
        #df_no_duplicates=df.drop_duplicates()
        #df_cleaned=df_no_duplicates.dropna()
        print("\n Performing Data transformation")
        print("\n Transformingcategorical object data into numeric format")
        le = LabelEncoder()
        for column in self.StringColumns:
            print("Transforming : ", column)
            label = le.fit_transform(self.dataframe[column])
            self.dataframe["trans_"+column] = label
        return self.dataframe

    def write_dataframe(self):
        print("Writing Dataframe to : ",self.writeFilePath)
        self.dataframe.to_csv(self.writeFilePath,index=False)



if __name__=="__main__":
    ami = AMI()
    #inputs 
    ami.readFilePath = "ProcessedData.csv"
    ami.StringColumns = ["type","nameOrig","nameDest"]
    '''
    object_columns=newdf.select_dtypes(include=['object'])
    print(object_columns.columns.to_list())
    '''
    ami.writeFilePath = "TransformedData.csv"
    #analysis
    ami.read_dataframe()
    ami.analysis_dataframe()
    ami.transform_dataframe_str2num()
    ami.write_dataframe()
    





