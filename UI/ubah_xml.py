import xml.etree.ElementTree as ET
 
mytree = ET.parse('label.xml')
myroot = mytree.getroot()
 
# iterating through the price values.
for label in myroot.iter('label'):
    # updates the price value
    label.text = str('mundur')
    # creates a new attribute
    #prices.set('newprices', 'yes')

mytree.write('label.xml')