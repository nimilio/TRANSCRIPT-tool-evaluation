#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-# LINE BY LINE EVAL #-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

import itertools
import editdistance
from sklearn.metrics import confusion_matrix
import pandas as pd

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-# Golden standard dictionaries, lists for lines #-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

Borg_Golden = {"Borg_0157r":[], 
              "Borg_0191r":[],
              "Borg_0201r":[]}

Borg_Golden_placeholders = {"Borg_0157r":[], 
              "Borg_0191r":[],
              "Borg_0201r":[]}

Borg_Golden_insecure = {"Borg_0157r":[], 
              "Borg_0191r":[],
              "Borg_0201r":[]}


Borg_Golden_set = set() 


def get_golden(cipher, cipher_dict, cipher_dict_insecure, cipher_dict_place):
    untranscribed = 0 
    uncertain = 0
    for page in cipher_dict: 
        with open("GT/"+cipher+page+".txt", "r") as file:
            for line in file: 
                line_list = [] 
                line_list_placeholders = []
                line_list_insecure = []
                strip = line.strip('\n') 
                no_paragraph = strip.split() 
                for char in no_paragraph: 
                    if char == " ": 
                        pass
                    elif char == "?": 
                        line_list_insecure.append(char)
                        line_list_placeholders.append(char)
                        line_list.append(char)
                        uncertain += 1
                    elif "?" in char and char != "??":
                        line_list_insecure.append(char)
                        longs = list(char) 
                        longs.remove("?") 
                        line_list.extend(longs) 
                        line_list_placeholders.extend(longs)
                        x = "".join(longs)
                        Borg_Golden_set.add(x)
                        uncertain += 1
                    elif char == "??":
                        untranscribed += 1
                        line_list_placeholders.append(char)
                        line_list_insecure.append(char)
                    else:
                        line_list_placeholders.append(char)
                        line_list_insecure.append(char)
                        line_list.append(char)
                        Borg_Golden_set.add(char)
                cipher_dict[page].append(line_list)
                cipher_dict_place[page].append(line_list_placeholders)
                cipher_dict_insecure[page].append(line_list_insecure)
                
    print ("Symbols in test but not in ground truth: ", untranscribed)
    print ("Uncertain cases: ", uncertain)



#-#-#-#-#-#-#-#-#-#-#-#-#
#-# Transcript  Tool  #-#
#-#-#-#-#-#-#-#-#-#-#-#-#

Borg_decrypt = {"Borg_0157r":[],
              "Borg_0191r":[],
              "Borg_0201r":[]}


Borg_decrypt_placeholders = {"Borg_0157r":[],
              "Borg_0191r":[],
              "Borg_0201r":[]}


Borg_decrypt_alphabet = set()



def decrypt_dicts(cipher_decrypt, folder, cipher_decrypt_place = None):
    missing_symbols = 0
    transcribed_symbols = 0
    untranscribed = 0

    for page in cipher_decrypt:
        with open("Decrypt/"+folder+"/"+page+".txt", "r") as file:
            for line in file:
                line_list = []
                line_list_placeholders = []
                strip = line.strip('\n')
                no_paragraph = strip.split()
                for char in no_paragraph:
                    if char == "?":
                        missing_symbols += 1
                        line_list.append(char)
                        if cipher_decrypt_place != None:
                            line_list_placeholders.append(char)
                    elif char == "SPACE":
                        line_list.append("<SPACE>")
                        Borg_decrypt_alphabet.add("<SPACE>")
                        if cipher_decrypt_place != None:
                            line_list_placeholders.append("<SPACE>")
                    elif cipher_decrypt_place != None and char == "??":
                        line_list_placeholders.append(char)
                        untranscribed += 1
                    else:
                        line_list.append(char)
                        Borg_decrypt_alphabet.add(char)
                        transcribed_symbols += 1
                        if cipher_decrypt_place != None:
                            line_list_placeholders.append(char)
                if not len(line_list) == 0:
                    cipher_decrypt[page].append(line_list)
                    if cipher_decrypt_place != None:
                        cipher_decrypt_place[page].append(line_list_placeholders)


    print("Missing symbols: ", str(missing_symbols)+" | " + str(round(missing_symbols/(transcribed_symbols+missing_symbols) *100,2))+"%")
    print("Untranscribed symbols: ", str(untranscribed)+" | "+ str(round(untranscribed/(transcribed_symbols+untranscribed+missing_symbols) *100,2))+ "%\n")




#-#-#-#-#-#-#-#-#-#
#-# EVALUATION  #-#
#-#-#-#-#-#-#-#-#-#

def character_recognition(cipher_dict, golden_dict, alphabet, gold_alphabet): 
    print("Character # recognized per page:")
    pages = 0
    total = 0
    recognized = 0
    for page1, page2 in zip(cipher_dict, golden_dict): 
        cipher_len = 0
        pages += 1
        gold_len = 0
        for line1, line2 in itertools.zip_longest(cipher_dict[page1], golden_dict[page2], fillvalue=list()): # 
            gold_len += len(line2) 
            for char in line1: #
                if char != "?":
                    cipher_len += 1
                    recognized += 1
        total += gold_len 
        print(page1+":", str(cipher_len)+"/"+str(gold_len), str(round(100*cipher_len/gold_len,2))+"%")
    print("\nTotal recognition =", str(round(recognized/total *100,2))+"% | ", str(recognized)+"/"+str(total))
    print("\nCharacters recognized:", str(len(alphabet))+"/"+str(len(gold_alphabet)), sorted(alphabet)) 
    print("\nFAILSAFE: Golden Alphabet=", sorted(gold_alphabet))
    print("\nAlphabet recognition: " , len(alphabet)*100 /len(gold_alphabet))
    print("\n - - Done - - -\n")



def evaluation(cipher_dict, golden_dict, insecure_dict = None):
    character_frequency = dict()
    golden_character_frequency = dict() 
    true_positives = dict() 
    accuracy_num = 0
    accuracy_den = 0
    gold_insecurities_match = []
    test_insecurities_match = dict()
    common_insecurities = []
    placeholders_comp = dict()
    placeholders_gold = dict()
    for page1, page2 in zip(cipher_dict, golden_dict):
        for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]): 
            for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'): 
                character_frequency[char1] = character_frequency.get(char1, 0)+1 
                golden_character_frequency[char2] = golden_character_frequency.get(char2, 0)+1
                if char1 == char2:
                    true_positives[char1] = true_positives.get(char1, 0)+1
                    accuracy_num += 1
                if char1 == "??": 
                    placeholders_comp[char2] = placeholders_comp.get(char2, 0)+1 
                if char2 == "??":
                    placeholders_gold[char1] = placeholders_gold.get(char1,0)+1
    if insecure_dict:
        for page1, page2 in zip(cipher_dict, insecure_dict):
            for line1, line2 in zip(cipher_dict[page1], insecure_dict[page2]):
                for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                    if char1 == '?':
                        test_insecurities_match[char2] = test_insecurities_match.get(char2, 0) +1
                    if '?' in char2 and char2.split('?')[0] == char1:
                        common_insecurities.append(tuple((char1,char2)))
                    if char1 == char2 and char1 == "?":
                        common_insecurities.append(tuple((char1,char2)))
                    if '?' in char2  and char2 != "??": 
                        gold_insecurities_match.append(tuple((char2,char1)))

                        
    for char in golden_character_frequency:
        accuracy_den += golden_character_frequency.get(char) 
    print("### Model Accuracy ###")
    print(str(round(accuracy_num*100/accuracy_den,2))+"%")
    print("")
    print("### Model Error Rate ###")
    print(str(round((1-accuracy_num/accuracy_den)*100,2))+"%")
    print("")
    print("### Precision/Recall/F1 value for each character ###\n")

    mean_precision = 0
    mean_recall = 0
    count = 0

    for character in true_positives:
        precision = true_positives[character] / character_frequency[character] 
        recall = true_positives[character] / golden_character_frequency[character] 
        mean_precision += precision
        mean_recall += recall
        count += 1
        print(character, "- P", round(precision, 4), "/ R", round(recall, 4), "/ Error Rate", round((1-precision), 4),"F1 scores", round(2*(recall*precision)/(recall+precision),2)) 
    mean_p = mean_precision*100/count 
    mean_r = mean_recall*100/count

    print("Mean Precision -", str(round(mean_p, 2))+"%")
    print("Mean Recall -", str(round(mean_r, 2))+"%")
    print("Avg error rate (chars) -", str(round(100-mean_p,2))+"%") 
    print("F1 score -", str(round(2*(mean_r*mean_p)/(mean_r+mean_p),2))+"%")

    print("\nCommon insecurities\n",common_insecurities)
    print("\nGold insecurities|Test matches\n",gold_insecurities_match)
    print("\nTest untranscribed|Gold matches\n",{k: v for k, v in sorted(test_insecurities_match.items(), key=lambda item: item[1], reverse = True)})

    print("\nPlaceholders in test set(symbols that were not recognized at all)\n", {k: v for k, v in sorted(placeholders_comp.items(), key=lambda item: item[1], reverse = True)})
    print("\nPlaceholders in gold set(symbols existing in test transcription but not in gold set)\n", {k: v for k, v in sorted(placeholders_gold.items(), key=lambda item: item[1], reverse = True)}, "\n")

    
    sorted_true_positive_frequencies = ({k: v for k, v in sorted(true_positives.items(), key=lambda item: item[1], reverse = True)})
    sorted_golden_character_frequencies = {k: v for k, v in sorted(golden_character_frequency.items(), key=lambda item: item[1], reverse = True)}


    normalized_freq= dict()
    for key in true_positives:
        if key in golden_character_frequency: 
            normalized_freq[key] = true_positives[key]*100/golden_character_frequency[key]


    print("True positives frequencies:")
    print(sorted_true_positive_frequencies)
    print("\n")
    print("Gold frequencies:")
    print(sorted_golden_character_frequencies)
    print("\n")
    print("Normalized frequencies:")
    print({k: v for k, v in sorted(normalized_freq.items(), key=lambda item: item[1], reverse = True)})


    # ________________________________________________________
    # Find the types of symbols that constitute this dataset |
    # ________________________________________________________

    print("All gold symbols: ", accuracy_den)
    print(sorted_golden_character_frequencies)
    print(character_frequency)
    digits = 0
    punc = 0
    alph = 0
    zodiac = 0
    alch = 0
    spaces = 0
    for key,value in sorted_golden_character_frequencies.items():
        if key == "4" or key == "5" or key == "6" or key == "8" or key == "2":
            digits += value
        if key == "," or key == "x":
            punc += value
        if key == "c" or key == "d" or key == "h":
            alph += value
        if key == "i" or key == "m" or key == "n" or key == "y" or key == "9" or key == "M" or key == "v":
            zodiac += value
        if key == "o" or key == "1" or key == "0" or key == "q" or key == "w":
            alch += value
        if key == "<SPACE>":
            spaces += value
    total_digits = digits/930 # 930 is accuracy_den - symbols in test but not in ground truth
    total_punc = punc/930
    total_alph = alph/930
    total_zodiac = zodiac/930
    total_alch = alch/930
    total_spaces = spaces/930
    print("Digits: " ,round(total_digits,4))
    print("Punctuation: " ,round(total_punc,4))
    print("Alphabet: " ,round(total_alph,4))
    print("Zodiac: " ,round(total_zodiac,4))
    print("Alchemical: " ,round(total_alch,4))
    print("Spaces: " ,round(total_spaces,4))
    #print(total_digits+total_alph+total_spaces+total_punc+total_alch+total_zodiac)

    print("\n - - Done - - -\n")


def ser(cipher_dict, golden_dict):
    tot_ser = 0
    tot_norm_ser = 0
    lines = 0
    for page1, page2 in zip(cipher_dict, golden_dict):
        for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
            truth = ''.join(line2) # remove spaces from the gold standard
            if "?" in line1:
                unkn = 0
                for char in line1:
                    if char == "?":
                        unkn +=1
                for i in range(unkn):
                    line1.remove("?")
            pred = ''.join(line1) # remove spaces from the test set
            SER = editdistance.eval(truth, pred) / len(truth) # N is the number of symbols in the reference text (ground truth)
            norm_SER = editdistance.eval(truth, pred) / max(len(truth), len(pred)) #https://towardsdatascience.com/evaluating-ocr-output-quality-with-character-error-rate-cer-and-word-error-rate-wer-853175297510
            tot_ser += SER
            tot_norm_ser += norm_SER
            lines += 1
    avg_ser = tot_ser / lines #mean value of all pages
    avg_norm_ser = tot_norm_ser / lines
    print("Avg. SER:", str(round(avg_ser*100,2))+"%") 
    print("Avg. SER, normalized:", str(round(avg_norm_ser*100,2))+"%")
    print("\n - - Done - - -\n")




def top_five_matrix_borg(cipher_dict, golden_dict, normalized = None):
    
    if normalized == True:
        keys = {"i" : "Gemini",
        "8" : "Eight",
        "q" : "MercurySublimate", 
        "x" : "Asterisk",
        "w" : "Arsenic", 
        "??" : "??",
        "d" : "Fire",
        "o" : "Square",
        "6" : "Six",
        "2" : "Two",
        "5" : "Five",
        "h" : "CapitalEta",
        "," : "Comma",
        "4" : "Four"}

        actual = list()
        predicted = list()
        for page1, page2 in zip(cipher_dict, golden_dict):
            for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
                for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                    if char2 == 'i' or char2 == '8' or char2 =='q' or char2 == 'x' or char2 == 'w':
                        actual.append(keys[char2])
                        if char1 == 'i' or char1 == '8' or char1 =='q' or char1 == 'x' or char1 == 'w':
                            predicted.append(keys[char1])
                        elif char1 != 'i' or char1 != '8' or char1 !='q' or char1 != 'x' or char1 != 'w':
                            predicted.append(keys[char1])
     
                        

        labels = [v for v in keys.values()]
        matrix = confusion_matrix(actual,predicted,labels = labels)
        frame = pd.DataFrame(matrix, index=labels, columns=labels)
    else:
        keys = {"w" : "Arsenic", 
        "x" : "Asterisk", 
        "6" : "Six",
        "<SPACE>" : "SPACE", 
        "h" : "CapitalEta", 
        "Other" : "Other"}

        actual = list()
        predicted = list()
    
        for page1, page2 in zip(cipher_dict, golden_dict):
            for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
                for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                    if char2 == 'w' or char2 == 'x' or char2 =='6' or char2 == 'h' or char2 == '<SPACE>':
                        actual.append(keys[char2])
                        if char1 == 'w' or char1 == 'x' or char1 =='6' or char1 == 'h' or char1 == '<SPACE>':
                            predicted.append(keys[char1])
                        else:
                            predicted.append("Other")
        actual.append("Other")
        predicted.append("Other")
        labels = [i for i in keys.values()]
        matrix = confusion_matrix(actual,predicted,labels = labels)
        frame = pd.DataFrame(matrix, index=labels, columns=labels)
    frame.to_excel("output_best.xlsx")
    print("\n - - Matrix Done - - -\n")




def worst_five_matrix_borg(cipher_dict, golden_dict, normalized = None):


    if normalized == True:
        keys = {"," : "Comma", 
        "<SPACE>" : "SPACE",
        "c" : "Delta", 
        "M" : "Virgo",
        "6" : "Six",
        "0" : "CopperOre",
        "w" : "Arsenic",
        ":" : "Colon",
        "." : "Dot",
        "o" : "Square",
        "??" : "??",
        "m" : "Scorpio",
        "q" : "MercurySublimate", 
        "1" : "IronOre",
        "?" : "?",
        "5" : "Five",
        "h" : "CapitalEta",
        "2" : "Two"}

        actual = list()
        predicted = list()
        for page1, page2 in zip(cipher_dict, golden_dict):
            for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
                for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                    if char2 == ',' or char2 == 'c' or char2 =='M' or char2 == '6' or char2 ==  "<SPACE>":
                        actual.append(keys[char2])
                        if char1 == ',' or char1 == 'c' or char1 =='M' or char1 == '6' or char1 == "<SPACE>":
                            predicted.append(keys[char1])
                        elif char1 != ',' or char1 != 'c' or char1 !='M' or char1 != '6' or char1 != 'w':
                            predicted.append(keys[char1])
        labels = [v for v in keys.values()]
        matrix = confusion_matrix(actual,predicted,labels = labels)
        frame = pd.DataFrame(matrix, index=labels, columns=labels)
    else:
        keys = {"y" : "Aquarius", 
        "2" : "Two",
        "," : "Comma", 
        "5" : "Five",
        "c" : "Delta",
        "Other" : "Other"}

        actual = list()
        predicted = list()
        for page1, page2 in zip(cipher_dict, golden_dict):
            for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
                for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                    if char2 == 'y' or char2 == '2' or char2 ==',' or char2 == '5' or char2 ==  "c":
                        actual.append(keys[char2])
                        if char1 == 'y' or char1 == '2' or char1 ==',' or char1 == '5' or char1 == "c":
                            predicted.append(keys[char1])
                        else:
                            predicted.append("Other")
        actual.append("Other")
        predicted.append("Other")
        labels = [v for v in keys.values()]
        matrix = confusion_matrix(actual,predicted,labels = labels)
        frame = pd.DataFrame(matrix, index=labels, columns=labels)
    frame.to_excel("output_worst.xlsx")
    print("\n - - Matrix Done - - -\n")


def cross_borg(cipher_dict, golden_dict):

    keys = { "8": "8",
    "2" : "2",
    "6" : "6",
    "4" : "4", 
    "5" : "5",
    "c" : "δ",
    "0" : "♀",
    "1" : "♂",
    "??" : "X",
    "?" : "?",
    "q" : "q",
    "o" : "o",
    "h" : "h",
    "9" : "9"
    }

    actual = list()
    predicted = list()
    for page1, page2 in zip(cipher_dict, golden_dict):
        for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
            for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                if char2 == '8' or char2 == '2' or char2 =='6' or char2 == '4' or char2 ==  "5"  or char2 ==  "c" or char2 ==  "0" or char2 ==  "1":
                    actual.append(keys[char2])
                    if char1 == '8' or char1 == '2' or char1 =='6' or char1 == '4' or char1 == "5" or char1 ==  "c" or char1 ==  "0" or char1 ==  "1":
                        predicted.append(keys[char1])
                    elif char1 != '8' or char1 != '2' or char1 !='6' or char1 != '4' or char1 != '5'  or char1 != 'c'  or char1 != '0'  or char1 != '1':
                        predicted.append(keys[char1])
    labels = [v for v in keys.values()]
    matrix = confusion_matrix(actual,predicted,labels = labels)
    frame = pd.DataFrame(matrix, index=labels, columns=labels)

    frame.to_excel("output_cross.xlsx")
    print("\n - - Matrix Done - - -\n")

##############################
### FIRST ROUND OF TESTING ###
###      OPTIMIZATION      ###
##############################

get_golden("Borg_Golden/", Borg_Golden, Borg_Golden_insecure, Borg_Golden_placeholders)
decrypt_dicts(Borg_decrypt, "Borg_5shot_4trsh/")
ser(Borg_decrypt, Borg_Golden)
decrypt_dicts(Borg_decrypt, "Borg_5shot_6trsh/")
ser(Borg_decrypt, Borg_Golden)
decrypt_dicts(Borg_decrypt, "Borg_5shot_8trsh/")
ser(Borg_decrypt, Borg_Golden)
decrypt_dicts(Borg_decrypt, "Borg_Omniglot")
ser(Borg_decrypt, Borg_Golden)



###############################
### SECOND ROUND OF TESTING ###
###############################

decrypt_dicts(Borg_decrypt, "Borg_5shot_4trsh/")
character_recognition(Borg_decrypt, Borg_Golden, Borg_decrypt_alphabet, Borg_Golden_set)
get_golden("Borg_Golden/placeholders/", Borg_Golden, Borg_Golden_insecure, Borg_Golden_placeholders)
decrypt_dicts(Borg_decrypt, "placeholders/Borg_5shot_4trsh/", Borg_decrypt_placeholders)
evaluation(Borg_decrypt_placeholders, Borg_Golden_placeholders, Borg_Golden_insecure)
top_five_matrix_borg(Borg_decrypt_placeholders, Borg_Golden_placeholders, False)
worst_five_matrix_borg(Borg_decrypt_placeholders, Borg_Golden_placeholders, False)
cross_borg(Borg_decrypt_placeholders, Borg_Golden_placeholders)
