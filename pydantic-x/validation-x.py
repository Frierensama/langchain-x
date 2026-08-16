from pydantic import BaseModel, Field, AnyUrl, EmailStr, field_validator, model_validator, computed_field
from typing import Optional, List, Dict, Annotated


class Patient(BaseModel):
    name:Annotated[str, Field(description='name of the person')]
    age:Annotated[int,Field(gt=0,lt=120,description='age of the person')]
    email:EmailStr
    contacts: Dict[str,str]

    # validating email field to be specific domains
    @field_validator('email')
    @classmethod
    def email_validator(cls, value):
        domain_ = value.split('@')[-1]
        if domain_ not in ['arc.daemon','gmail.com','chloe.riruru']:
            raise ValueError('invalid mail')
        return value

    # model validation - if a feild depends on other field/s
    @model_validator(mode='after') # mode tells validate this after [type coersion '32' is 32]
    def emergency_contact_validator(self):
        if (self.age <18 or self.age >60) and 'emergency' not in self.contacts:
            raise ValueError('this age patient requires an emergency contact')
        return self

    @computed_field
    @property
    def total_contacts(self)->int:
        return len(self.contacts)

def print_patient(patient:Patient):
    print(patient.name)
    print(patient.age)
    print(patient.email)
    print(patient.contacts)
    print(patient.total_contacts)


patient1_info = {
    'name':'rudy',
    'age':21,
    'email':'rudeusgreyray@gmail.com',
    'contacts':{'self':'002'}
}
patient2_info = {
    'name':'clove',
    'age':17,
    'email':'akai@chloe.riruru',
    'contacts':{'self':'1618','emergency':'02'}
}
patient3_info = {
    'name':'diablo',
    'age':23,
    'email':'diablo@arc.daemon',
    'contacts':{'self':'02'}
}

patient1 = Patient(**patient1_info)
patient2 = Patient(**patient2_info)
patient3 = Patient(**patient3_info)

print_patient(patient1)
print_patient(patient2)
print_patient(patient3)