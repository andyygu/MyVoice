import pandas as pd
import csv
import scipy
from scipy.stats import chi2_contingency

def is_worried(response):
    if response.find('Yes') != -1:
        return True
    # if response.find('Yea') != -1:
    #     return True
    # if response.find('not worried') != -1:
    #     return False
    if response.find('No') != -1:
        return False

covid1 = pd.read_csv('covid1.csv.csv')
covid1["Is worried"] = ''

prompt = "Are you worried about coronavirus? Why or why not?"

race_worried = covid1[["Race", prompt, "Is worried"]]

#print(race_worried)

for index, row in race_worried.iterrows():
    #print(type(row[prompt]))
    if type(row[prompt]) == float:
        row["Is worried"] = False
    else:
        if is_worried(row[prompt]):
            row["Is worried"] = True
        else:
            row["Is worried"] = False
    
#print(race_worried["Is worried"])

ai_worried = 0
ai_not_worried = 0
asian_worried = 0
asian_not_worried = 0
black_worried = 0
black_not_worried = 0
hawaiian_worried = 0
hawaiian_not_worried = 0
white_worried = 0
white_not_worried = 0
other_worried = 0
other_not_worried = 0
mixed_worried = 0
mixed_not_worried = 0

for index, row in race_worried.iterrows():
    if type(row["Race"]) != float:
        if row["Race"].find(",") != -1:
            if row["Is worried"]:
                mixed_worried += 1
            else:
                mixed_not_worried += 1
        elif row["Race"].find("American Indian or Alaska Native") != -1:
            if row["Is worried"]:
                ai_worried += 1
            else:
                ai_not_worried += 1
        elif row["Race"].find("Asian") != -1:
            if row["Is worried"]:
                asian_worried += 1
            else:
                asian_not_worried += 1
        elif row["Race"].find("Black or African American") != -1:
            if row["Is worried"]:
                black_worried += 1
            else:
                black_not_worried += 1
        elif row["Race"].find("Native Hawaiian or Other Pacific Islander") != -1:
            if row["Is worried"]:
                hawaiian_worried += 1
            else:
                hawaiian_not_worried += 1
        elif row["Race"].find("Other") != -1:
            if row["Is worried"]:
                other_worried += 1
            else:
                other_not_worried += 1
        elif row["Race"].find("White or Caucasian") != -1:
            if row["Is worried"]:
                white_worried += 1
            else:
                white_not_worried += 1
        
print("AI: ", ai_not_worried, " ", ai_worried)
print("Asian: ", asian_not_worried, " ", asian_worried)
print("Black: ", black_not_worried, " ", black_worried)
print("Hawaiian: ", hawaiian_not_worried, " ", hawaiian_worried)
print("White: ", white_not_worried, " ", white_worried)
print("Other: ", other_not_worried, " ", other_worried)
print("Mixed: ", mixed_not_worried, " ", mixed_worried)

#defining the table   
data = [[ai_not_worried,asian_not_worried,black_not_worried,hawaiian_not_worried,white_not_worried,other_not_worried,mixed_not_worried],[ai_worried,asian_worried,black_worried,hawaiian_worried,white_worried,other_worried,mixed_worried]]
stats,p,dof,expected = chi2_contingency(data)

#interpret p-value
alpha = 0.10
print("p value is: " + str(p))
if p <= alpha: 
    print('Dependent (reject H0)')
else: 
    print('Independent (H0 holds true)')
