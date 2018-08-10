import re

mapDict={
	'JioPay':{
		"type":'Telecom',
		"name":'JIO',
		"Category":['bill','Recharge']
	},
	'JIOPAY':{
		"type":'Telecom',
		"name":'JIO',
		"Category":['bill','Recharge']
	},
	'AxisBk':{
		"type":'Bank',
		"name":'AXIS BANK',
		"Category":['Credit Card']
	},
	'IPRUMF':{
		"type":'Bank',
		"name":'ICICI BANK',
		"Category":['Credit Card','SIP','EMI']
	},
	'BAJAJF':{
		"type":'Bank',
		"name":'BAJAJ Finance',
		"Category":['Credit Card','SIP','EMI']
	},
	'VFCARE':{
		"type":'Telecom',
		"name":'Vodafone',
		"Category":['bill','Recharge']
	},
	'HDFCBK':{
		"type":'Bank',
		"name":'HDFC BANK',
		"Category":['CreditCard']
	},
	'IDEA':{
		"type":'Telecom',
		"name":'Idea',
		"Category":['bill','Recharge']
	},
	'idea':{
		"type":'Telecom',
		"name":'Idea',
		"Category":['bill','Recharge']
	},
	'DishTV':{
		"type":'TV',
		"name":'Dish TV',
		"Category":['bill','Recharge']
	},
	'FICCLT':{
		"type":'Bank',
		"name":'Fullerton India',
		"Category":['Loan']
	},
	'LICIND':{
		"type":'Insurance',
		"name":'LIC',
		"Category":['Insurance']
	},
	'AIREXP':{
		"type":'Telecom',
		"name":'Airtel',
		"Category":['bill','Recharge']
	},
	'MYAMEX':{
		"type":'Bank',
		"name":'American Express',
		"Category":['Credit Card']
	},
	'ICICIB':{
		"type":'Insurance',
		"name":'LIC',
		"Category":['Mutual Fund']
	},
	'UnionB':{
		"type":'Bank',
		"name":'Union Bank',
		"Category":['Mutual Fund','Loan','EMI']
	},
	'TPOWER':{
		"type":'Power',
		"name":'TPOWER',
		"Category":['Electricity']
	},
	'BESCOM':{
		"type":'Power',
		"name":'BESCOM',
		"Category":['Electricity']
	},
	'POLBAZ':{
		"type":'Insurance',
		"name":'Policy Bazaar',
		"Category":['Insurance']
	},
	'MYTSKY':{
		"type":'Bill',
		"name":'TATA Sky',
		"Category":['DTH']
	},
	'AIRDTH':{
		"type":'Bill',
		"name":'Airtel DTH',
		"Category":['DTH']
	},
	'DTHVID':{
		"type":'Bill',
		"name":'Videocon DTH',
		"Category":['DTH']
	},
}

#recharge, payment,bill
impWords=['expire','payable','due','pending','awaited','overdue','fee','charge','EMI','installment']

def extract_date(text):

	x=re.findall(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|January|Februry|March|April|May|June|July|August|Sepember|October|November|December|\d{1,2})+(?:\w{2})*(?:-|\.|\s|,|\/|\-)+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|January|Februry|March|April|May|June|July|August|Sepember|October|November|December|\d{2,4})+(?:-|\.|\s|,|\/|\'|\-)*(?:\d{2,4})',text)
	date=None

	for i in range(len(x)):
		if x[i].find('-'):
			date=x[i]

		elif x[i].find(' ')  or x[i].find('/') :
			date= x[i]

	if date!=None :
		if len(date.split('.'))==2:
			date=None
		elif len(date.split('.'))==3 and len(date.split('.')[2]) not in [2,4]:
			date=None

	if not date:
		x=re.findall(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|January|Februry|March|April|May|June|July|August|Sepember|October|November|December|\d{1,2})+(?:-|\.|\s|,|\/|\-)+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|January|Februry|March|April|May|June|July|August|Sepember|October|November|December|\d{2,4})+(?:-|\.|\s|,|\/|\'|\-)+',text)
		if len(x)!=0 :
			date=x[0]
			if len(date.split('.'))==2:
				date=None
			elif len(date.split('.'))==3 and len(date.split('.')[2]) not in [2,4]:
				date=None
	
	return date

def hasNumbers(inputString):
	return any(char.isdigit() for char in inputString)

def extract_money(text):
	# return re.findall(r'(?:rs|RS|Rs)*(?:-|\.|\s)*(?:[\d.,]+)(?:-|\.|\s)*(?:rs|RS|Rs)*',text)
	x= re.findall(r'(?:rs|RS|Rs|INR|MRP)(?:-|\.|\s)*(?:[\d.,]+)',text)

	for k in x:
		if not hasNumbers(k):
			x.remove(k)

	if len(x)==0:
		x= re.findall(r'([\d\.,])+(?:-|\.|\s)*(?:rs|RS|Rs|INR|MRP)(\,)*',text)

	#special cases
	if 'DTHVID' in text and 'expired' in text and 'customer id' in text :
		x=['Rs. 200']
	elif 'JIOINF' in text and 'Jio' in text and 'expire on' in text :
		x=['Rs. 399']
	elif 'LICIND' in text and 'premium due' in text and 'policy' in text :
		x=['Rs. 0']

	if len(x)==0 :
		return None
	else:
		return x[0]

def checkSpecialCase(details,sms):
	#return True or False
	flag=True
	if "Jio" in sms:
		if '100%' not in sms:
			flag =False

	return flag

def checksms(x):
	details,sms=x.split(",",1)
	if '-' in details:
		y,details=details.split("-",1)

	#check if sms is promotional sms
	if details.isdigit():
		pass
	else:
		date=None
		money=None
		date=extract_date(x)
		money=extract_money(x)

		if set(sms.split()).intersection(set(impWords)) and money!=None:
			if 'OTP' not in sms and 'otp' not in sms and 'successful' not in sms and 'CIBIL' not in sms and 'Thank you for using' not in sms and 'Recharge done' not in sms and'received' not in sms and'paid online' not in sms :
				# print 'here'

				# if date!=None:# or money!=None:
				# if checkSpecialCase(details,sms):
				if details in mapDict.keys():
					print money,date,mapDict[details[-6:]]['type'],mapDict[details[-6:]]['name'],mapDict[details[-6:]]['Category']

		#special frequent cases
		if "Jio" in sms and '100%'  in sms:
			print money,date,mapDict[details[-6:]]['type'],mapDict[details[-6:]]['name'],mapDict[details[-6:]]['Category']
						
		elif 'Recharge dt for ' in sms and 'DishTV' in details:
			print 'Rs 0',date,mapDict[details[-6:]]['type'],mapDict[details[-6:]]['name'],mapDict[details[-6:]]['Category']
							
		


if __name__ == "__main__":

	with open('/home/yash/github/smsdata.txt') as f:
	    content = f.readlines()

	for x in content[:]:
		checksms(x)

	checksms("AX-BESCOM,Dear Consumer, your bill of Rs. 485.00 due on 2018-06-17 for Account ID 5129322104 has been generated. To view and pay online click www.bescom.co.in. Treat this SMS as 15 days ")

