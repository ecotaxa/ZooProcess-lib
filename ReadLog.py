# from configparser import ConfigParser 

# from ProjectClass import ProjectClass
from pathlib import Path


class ReadLog:

    def __init__(self, logfile:Path ) -> None:
        
        # self.TP = TP
        # sample = sample
        # self.logfile = TP.getLogFile(sample)
        self.logfile = logfile
        # self.configur = ConfigParser()
        # self.configur.read(logfile)

    def _find_key_in_file(self, key: str) -> str:
        print(f"Call _find_key_in_file({key})")
        with open(self.logfile) as f:
            # print(f"file: {f}")
            for line in f:
                # print(f"line: {line}")
                if key in line:
                    return line.split('=')[1].strip()
        return None

    def getBackgroundPattern(self) -> str:
        print(f"Call getBackgroundPattern()")

        # self.configur.read(self.logfile)
        # # background = self.configur['Image_Process']['Background_correct_using']
        # background = self.configur.get('Image_Process','Background_correct_using')
        # print ("Installation Background_correct_using : ", background) 
        background = self._find_key_in_file("Background_correct_using")
        print (F"Background_correct_using : {background}") 

        lenght = len("20141003_1144")
        return background[:lenght]

    # file = "t_17_2_tot_1_log.txt"



    # configur = ConfigParser()
    # print (configur.read('config.ini'))