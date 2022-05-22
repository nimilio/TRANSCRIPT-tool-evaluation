#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-# LINE BY LINE EVAL #-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

import itertools
import editdistance
from sklearn.metrics import confusion_matrix
import re
import pandas as pd

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-# Golden standard dictionaries, lists for lines #-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

Copiale_Golden = {"Copiale_003":[], 
              "Copiale_005":[],
              "Copiale_009":[],
              "Copiale_030":[], 
              "Copiale_039":[],
              "Copiale_079":[],
              "Copiale_096":[], 
              "Copiale_100":[]}


Copiale_Golden_placeholders = {"Copiale_003":[], 
              "Copiale_005":[],
              "Copiale_009":[],
              "Copiale_030":[], 
              "Copiale_039":[],
              "Copiale_079":[],
              "Copiale_096":[], 
              "Copiale_100":[]}



Copiale_Golden_set = set() 



def get_golden(cipher, cipher_dict, cipher_dict_place):
    untranscribed = 0 # exist in test but not in ground truth
    for page in cipher_dict:
        with open("GT/"+cipher+"/"+page+".txt", "r") as file:
            for line in file:
                line_list = []
                line_list_placeholders = []
                strip = line.strip('\n')
                no_paragraph = strip.split()
                for char in no_paragraph:
                    if char == " ": 
                        pass
                    elif char == "X":
                        untranscribed += 1
                        line_list_placeholders.append(char)
                    else:
                        line_list.append(char.lower())
                        line_list_placeholders.append(char.lower())
                        Copiale_Golden_set.add(char.lower())
                cipher_dict[page].append(line_list)
                cipher_dict_place[page].append(line_list_placeholders)


    print ("Symbols in test but not in ground truth: ", untranscribed)




#-#-#-#-#-#-#-#-#-#-#-#-#
#-# Transcript  Tool  #-#
#-#-#-#-#-#-#-#-#-#-#-#-#


Copiale_decrypt = {"Copiale_003":[], 
              "Copiale_005":[],
              "Copiale_009":[],
              "Copiale_030":[], 
              "Copiale_039":[],
              "Copiale_079":[],
              "Copiale_096":[], 
              "Copiale_100":[]}

Copiale_decrypt_placeholders = {"Copiale_003":[], 
              "Copiale_005":[],
              "Copiale_009":[],
              "Copiale_030":[], 
              "Copiale_039":[],
              "Copiale_079":[],
              "Copiale_096":[], 
              "Copiale_100":[]}



Copiale_decrypt_alphabet = set()



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
                dots = strip.replace(". . .","...")
                no_paragraph = dots.split()
                for char in no_paragraph:
                    if char == "?":
                        missing_symbols += 1
                        line_list.append(char)
                        if cipher_decrypt_place != None:
                            line_list_placeholders.append(char)
                    elif char == ". . .":
                        line_list.append("...")
                        Copiale_decrypt_alphabet.add("...")
                        transcribed_symbols += 1
                        if cipher_decrypt_place != None:
                            line_list_placeholders.append("...")
                    elif cipher_decrypt_place != None and char == "X":
                        line_list_placeholders.append(char)
                        untranscribed += 1
                    else:
                        line_list.append(char)
                        Copiale_decrypt_alphabet.add(char)
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

def character_recognition(cipher_dict, golden_dict, alphabet, gold_alphabet): #alphabet:the tested one
    print("Character # recognized per page:")
    pages = 0
    total = 0
    recognized = 0
    for page1, page2 in zip(cipher_dict, golden_dict): # page1 and page2 are the key names of the dictionaries
        cipher_len = 0
        pages += 1
        gold_len = 0
        for line1, line2 in itertools.zip_longest(cipher_dict[page1], golden_dict[page2], fillvalue=list()): # 
            gold_len += len(line2) # how many elements has the list (which are the symbols included in a line) of the gold set
            for char in line1: # this for loop actually does the same work as the previous line but for the test set
                if char != "?":
                    cipher_len += 1
                    recognized += 1
        total += gold_len # total includes the number of symbols from all pages of the golden set
        print(page1+":", str(cipher_len)+"/"+str(gold_len), str(100*cipher_len/gold_len)+"%")
    print("\nTotal recognition =", str(round(recognized/total *100,2))+"% | ", str(recognized)+"/"+str(total))
    print("\nCharacters recognized:", str(len(alphabet))+"/"+str(len(gold_alphabet)), sorted(alphabet)) #alphabet comparison
    print("\nFAILSAFE: Golden Alphabet=", sorted(gold_alphabet))
    print("\nAlphabet recognition: " , len(alphabet)*100 /len(gold_alphabet))
    print("\n - - Done - - -\n")



def evaluation(cipher_dict, golden_dict):
    character_frequency = dict()
    golden_character_frequency = dict() 
    true_positives = dict() 
    accuracy_num = 0
    accuracy_den = 0
    test_insecurities_match = dict()
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
                if char1 == "X": 
                    placeholders_comp[char2] = placeholders_comp.get(char2, 0)+1 
                if char2 == "X":
                    placeholders_gold[char1] = placeholders_gold.get(char1,0)+1
                if char1 == '?':
                        test_insecurities_match[char2] = test_insecurities_match.get(char2, 0) +1
                        
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

    print("All gold symbols ", accuracy_den)
    print(sorted_golden_character_frequencies)
    print(character_frequency)
    punc = 0
    alph = 0
    digits = 0
    alch = 0
    misc = 0
    for key,value in sorted_golden_character_frequencies.items():
        if key == "three":
            digits += value
        if key == ":" or key == "." or key == '"' or key == "...":
            punc += value
        if key == "hd" or key == "tri" or key == 'fem' or key == "mal":
            alch += value
        if key == "i" or key == "c" or key == 'ns' or key == "v" or key == "z" or key == 'uh' or key == "g" or key == "uu" or key == 'j' or key == "d" or key == "a" or key == 'b' or key == "n" or key == 'o.' or key == "l" or key == "pi" or key == 'oh' or key == "eh" or key == "r." or key == 'p' or key == "s." or key == "r" or key == 'gam' or key == "ru" or key == "ih" or key == 'h.' or key == "lam" or key == "del" or key == 'e' or key == "y.." or key == "ah" or key == "p." or key == 'nu' or key == "ds" or key == "mu" or key == 'h' or key == "m." or key == "x." or key == 'n.' or key == "k" or key == "o" or key == 'm' or key == "u" or key == "f" or key == 'iot' or key == "s" or key == "zs" or key == 't' or key == "w" or key == "gs" or key == "x" or key == "y" or key == "q" or key == "c.": 
            alph += value
        if key == "ni" or key == "ki" or key == 'tri..' or key == "lip" or key == "nee" or key == 'o..' or key == "star" or key == "bigx" or key == 'gat' or key == "toe" or key == "arr" or key == 'bas' or key == "car" or key == "plus" or key == 'cross' or key == "ft" or key == "no" or key == 'sqp' or key == "zzz" or key == "pipe" or key == 'longs' or key == "grr" or key == "grl" or key == 'grc' or key == 'hk' or key == "sqi" or key == "bar" or key == 'inf'  or key == "qua":
            misc += value
    total_digits = digits/5670 # 5670 is accuracy_den - symbols in test but not in ground truth
    total_punc = punc/5670
    total_alch = alch/5670
    total_alph = alph/5670
    total_misc = misc/5670
    print("Digits: " ,total_digits)
    print("Alphabet: " ,total_alph)
    print("Alchemical: " ,total_alch)
    print("Miscellaneous: " ,total_misc)
    print("Punctuation: " ,total_punc)
    #print(total_digits+total_alph+total_misc+total_punc+total_alch)
    print("\n - - Done - - -\n")



def ser(cipher_dict, golden_dict):
    tot_ser = 0
    tot_norm_ser = 0
    lines = 0
    for page1, page2 in zip(cipher_dict, golden_dict):
        for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
            truth = ''.join(line2) 
            if "?" in line1:
                unkn = 0
                for char in line1:
                    if char == "?":
                        unkn +=1
                for i in range(unkn):
                    line1.remove("?")
            pred = ''.join(line1) 
            SER = editdistance.eval(truth, pred) / len(truth)
            norm_SER = editdistance.eval(truth, pred) / max(len(truth), len(pred))
            tot_ser += SER
            tot_norm_ser += norm_SER
            lines += 1
    avg_ser = tot_ser / lines #mean value of all pages
    avg_norm_ser = tot_norm_ser / lines
    print("Avg. SER:", str(round(avg_ser*100,2))+"%") 
    print("Avg. SER, normalized:", str(round(avg_norm_ser*100,2))+"%")
    print("\n - - Done - - -\n")



def top_five_matrix_copiale(cipher_dict, golden_dict, normalized = None):
    

    if normalized == True:
        keys = {"pi" : "π", 
        "p" : "p",
        "ni" : "=", 
        "plus" : "+",
        "c" : "c",
        "f" : "f",
        "sqi" : "sqi",
        "X" : "X",
        "nu" : "ṉ",
        "r" : "r",
        "b" : "b",
         "?" : "?",
        "eh" : "ê"}


        actual = list()
        predicted = list()
        for page1, page2 in zip(cipher_dict, golden_dict):
            for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
                for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                    if char2 == 'pi' or char2 == 'p' or char2 =='ni' or char2 == 'plus' or char2 == 'c':
                        actual.append(keys[char2])
                        if char1 == 'pi' or char1 == 'p' or char1 =='ni' or char1 == 'plus' or char1 == 'c':
                            predicted.append(keys[char1])
                        elif char1 != 'pi' or char1 != 'p' or char1 !='ni' or char1 != 'plus' or char1 != 'c':
                            predicted.append(keys[char1])

        labels = [v for v in keys.values()]
        matrix = confusion_matrix(actual,predicted,labels = labels)
        frame = pd.DataFrame(matrix, index=labels, columns=labels)
    else:
        keys = {"lam" : "Λ", 
        "uu" : "ṵ", 
        "z" : "z", 
        "c." : "ċ", 
        "bar" : "|",
        "Other" : "Other"}


        actual = list()
        predicted = list()
    
        for page1, page2 in zip(cipher_dict, golden_dict):
            for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
                for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                    if char2 == 'lam' or char2 == 'uu' or char2 =='z' or char2 == 'c.' or char2 == 'bar':
                        actual.append(keys[char2])
                        if char1 == 'lam' or char1 == 'uu' or char1 =='z' or char1 == 'c.' or char1 == 'bar':
                            predicted.append(keys[char1])
                        else:
                            predicted.append("Other")

        actual.append("Other")
        predicted.append("Other")
        labels = [i for i in keys.values()]
        matrix = confusion_matrix(actual,predicted,labels = labels)
        frame = pd.DataFrame(matrix, index=labels, columns=labels)
        print(set(unknown))
    frame.to_excel("output_best.xlsx")
    print("\n - - Matrix Done - - -\n")
        




ef worst_five_matrix_copiale(cipher_dict, golden_dict, normalized = None):


    if normalized == True:
        keys = {"..." : "...", 
        ":" : ":",
        "iot" : "ι", 
        "." : ".",
        "sqi" : "sqi",
        "sqp" : "sqp",
        "uu" : "ṵ",
        "p" : "p",
        "ru" : "ṟ",
        "bas" : "bas",
        "z" : "z",
        "k" : "k",
        "x." : "ẋ",
        "zzz": "zzz",
        "cross": "✝",
        "g" : "g",
        "?" : "?",
        "hk" : "hk",
        "del" : "δ",
        "j" : "İ",
        "u" : "u",
        "bar" : "|",
        "hd" : "ħ",
        "s": "s",
        "n" : "n",
        "plus" : "+",
        "lam" : "Λ",
        "X" : "X",
        "m": "m",
        "(.|.|iot|iot|.)" : "(.|.|iot|iot|.)",
        "(:|:|:|:)" : "(:|:|:|:)",
        "(...|:|:|ds)" : "(...|:|:|ds)",
        "(.|:|.|.|x.)" : "(.|:|.|.|x.)"}

        actual = list()
        predicted = list()
        for page1, page2 in zip(cipher_dict, golden_dict):
            for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
                for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                    if char2 == '...' or char2 == ':' or char2 =='iot' or char2 == '.' or char2 ==  "sqi":
                        actual.append(keys[char2])
                        if char1 == '...' or char1 == ':' or char1 =='iot' or char1 == '.' or char1 == "sqi":
                            predicted.append(keys[char1])
                        elif char1 != '...' or char1 != ':' or char1 !='iot' or char1 != '.' or char1 != "sqi":
                            predicted.append(keys[char1])


        labels = [v for v in keys.values()]
        matrix = confusion_matrix(actual,predicted,labels = labels)
        frame = pd.DataFrame(matrix, index=labels, columns=labels)
    else:
        keys = {"..." : "...", 
        "car" : "car",
        "tri.." : "tri..", 
        "." : ".",
        "nee" : "nee",
        "Other": "Other"}

        actual = list()
        predicted = list()
        for page1, page2 in zip(cipher_dict, golden_dict):
            for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
                for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                    if char2 == '...' or char2 == 'car' or char2 =='tri..' or char2 == '.' or char2 ==  "nee":
                        actual.append(keys[char2])
                        if char1 == '...' or char1 == 'car' or char1 =='tri..' or char1 == '.' or char1 == "nee":
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


def cross_copiale(cipher_dict, golden_dict):

    keys = { "three": "3",
    ":" : ":",
    "del" : "δ",
    "mal" : "mal",
    "fem" : "♀",
    "..." : "...",
    "gs" : "gs",
    "arr" : "arr",
    "." : ".",
    "c." : "c.",
    "bar" : "|",
    "X" : "X",
    "ah" : "ah",
    "p" : "p",
    "j" : "j",
    "n." : "n.",
    "iot" : "iot",
    "lam" : "lam",
    "g" : "g",
    "ns" : "ns",
    "ih" : "ih",
    "sqi" : "sqi",
    "oh" : "oh",
    "bas" : "bas",
    "plus" : "plus",
    "ru" : "ru",
    "sqp" : "sqp",
    "zzz" : "zzz",
    "uu" : "uu",
    "d" : "d",
    "n" : "n",
    "o" : "o",
    "u" : "u",
    "c" : "c",
    "x.": "x.",
    "uh" : "uh",
    "z" : "z",
    "k" : "k",
    "hd" : "hd",
    "s" : "s",
    "h." : "h.",
    "v" : "v",
    "eh" : "eh",
    "tri" : "tri"
    }

    actual = list()
    predicted = list()
    for page1, page2 in zip(cipher_dict, golden_dict):
        for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
            for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                if char2 == 'three' or char2 == ':' or char2 =='del' or char2 == 'mal' or char2 ==  "fem" :
                    actual.append(keys[char2])
                    if char1 == 'three' or char1 == ':' or char1 =='del' or char1 == 'mal' or char1 == "fem":
                        predicted.append(keys[char1])
                    elif char1 != 'three' or char1 != ':' or char1 !='del' or char1 != 'mal' or char1 != 'fem':
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

get_golden("Copiale_Golden", Copiale_Golden, Copiale_Golden_placeholders)
decrypt_dicts(Copiale_decrypt, "Copiale_5shot_4trsh")
ser(Copiale_decrypt, Copiale_Golden)
decrypt_dicts(Copiale_decrypt, "Copiale_5shot_6trsh")
ser(Copiale_decrypt, Copiale_Golden)
decrypt_dicts(Copiale_decrypt, "Copiale_5shot_8trsh")
ser(Copiale_decrypt, Copiale_Golden)
decrypt_dicts(Copiale_decrypt, "Copiale_Omniglot")
ser(Copiale_decrypt, Copiale_Golden)


###############################
### SECOND ROUND OF TESTING ###
###############################

decrypt_dicts(Copiale_decrypt, "Copiale_5shot_4trsh")
character_recognition(Copiale_decrypt, Copiale_Golden, Copiale_decrypt_alphabet, Copiale_Golden_set)
get_golden("Copiale_Golden/placeholders/", Copiale_Golden, Copiale_Golden_placeholders)
decrypt_dicts(Copiale_decrypt, "placeholders/Copiale_5shot_4trsh/", Copiale_decrypt_placeholders)
evaluation(Copiale_decrypt_placeholders, Copiale_Golden_placeholders)
top_five_matrix_copiale(Copiale_decrypt_placeholders, Copiale_Golden_placeholders, True)
worst_five_matrix_copiale(Copiale_decrypt_placeholders, Copiale_Golden_placeholders, True)
cross_copiale(Copiale_decrypt_placeholders, Copiale_Golden_placeholders)

