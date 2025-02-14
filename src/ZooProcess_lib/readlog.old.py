from configparser import ConfigParser 

from ZooscanProject import ZooscanProject

# Malheureusement ne fonctionne pas
# car le fichier n'est pas un vrai config file :(   

class ReadLog:

    def __init__(self, TP: ZooscanProject, sample:str) -> None:
        
        self.TP = TP
        # sample = sample
        logfile = self.TP.getLogFile(sample)

        self.configur = ConfigParser()
        self.configur.read(logfile)

    def getBackgroundPattern(self) -> str:
        self.configur.read(self.logfile)
        # background = self.configur['Image_Process']['Background_correct_using']
        background = self.configur.get('Image_Process','Background_correct_using')
        print ("Installation Background_correct_using : ", background) 

        return background
        


    # file = "t_17_2_tot_1_log.txt"



    # configur = ConfigParser()
    # print (configur.read('config.ini'))