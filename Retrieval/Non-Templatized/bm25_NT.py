 
from __future__ import absolute_import 
import openai
from openai.error import RateLimitError, InvalidRequestError
import json
import os
import sqlite3
import re
import numpy as np
from tqdm import tqdm, trange
import time
import sqlparse
 
import torch # the main pytorch library
import torch.nn.functional as f
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import pickle
import csv

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

nltk.download('punkt')
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
path_to_database_directory_dev = f'/path/to/dev/databases'
 
print(f'here line 33')
cache_dir = 'all-mpnet-base-v2'  # You can use different models here
recall_1 = 10
recall_2 = 10
 
folder='TP'
 
print(f'here line 42')
sys_prompt_fin = f'''You are a database administrator and have designed the following database for Financial whose schema is represented as:

CREATE TABLE account(
	account_id INTEGER PRIMARY KEY,
	district_id INTEGER,
	frequency TEXT,
	date DATE
);

CREATE TABLE card(
	card_id INTEGER PRIMARY KEY -- id number of credit card,
	disp_id INTEGER, -- disposition id
	type TEXT, -- type of credit card
	issued DATE, -- the date when the credit card issued 
);

CREATE TABLE client(
	client_id INTEGER PRIMARY KEY, -- the unique number
	gender TEXT,
	birth_date DATE,
	district_id INTEGER, -- location of branch
);

CREATE TABLE disp(
	disp_id INTEGER PRIMARY KEY, -- unique number of identifying this row of record
	client_id INTEGER, -- id number of client
	account_id INTEGER, -- id number of account
	type TEXT, -- type of disposition
);

CREATE TABLE district(
	district_id INTEGER PRIMARY KEY, -- location of branch
	A2 TEXT, -- district_name
	A3 TEXT, -- region
	A4 TEXT, -- number of inhabitants 
	A5 TEXT, -- no. of municipalities with inhabitants < 499
	A6 TEXT, -- no. of municipalities with inhabitants 500-1999
	A7 TEXT, -- no. of municipalities with inhabitants 2000-9999
	A8 INTEGER, -- no. of municipalities with inhabitants > 10000
	A9 INTEGER              ,
	A10 REAL, -- ratio of urban inhabitants
	A11 INTEGER, -- average salary
	A12 REAL, -- unemployment rate 1995
	A13 REAL, -- unemployment rate 1996
	A14 INTEGER, -- no. of entrepreneurs per 1000 inhabitants
	A15 INTEGER, -- no. of committed crimes 1995
	A16 INTEGER -- no. of committed crimes 1996
);

CREATE TABLE loan(
	loan_id INTEGER primary key, -- the id number identifying the loan data
	account_id INTEGER, -- the id number identifying the account
	date DATE, -- the date when the loan is approved
	amount INTEGER, -- approved amount
	duration INTEGER, -- loan duration
	payments REAL, -- monthly payments
	status TEXT, -- repayment status
	foreign key (account_id) references account (account_id)
);

CREATE TABLE `order`(
	order_id INTEGER primary key, -- identifying the unique order
	account_id INTEGER, -- id number of account
	bank_to TEXT, -- bank of the recipient
	account_to INTEGER, -- account of the recipient
	amount REAL, -- debited amount
	k_symbol TEXT, -- purpose of the payment
	foreign key (account_id) references account (account_id)
);

CREATE TABLE trans(
	trans_id INTEGER primary key, -- transaction id
	account_id INTEGER,
	date DATE, -- date of transaction
	type TEXT, -- +/- transaction. "PRIJEM" stands for credit, "VYDAJ" stands for withdrawal
	operation TEXT, -- mode of transaction. "VYBER KARTOU": credit card withdrawal, "VKLAD": credit in cash, "PREVOD Z UCTU":collection from another bank, "VYBER": withdrawal in cash, "PREVOD NA UCET": remittance to another bank
	amount INTEGER, -- amount of money. Unit：USD
	balance INTEGER, -- balance after transaction. Unit：USD
	k_symbol TEXT, -- characterization of the transaction. "POJISTNE": stands for insurrance payment, "SLUZBY": stands for payment for statement, "UROK": stands for interest credited, "SANKC. UROK": sanction interest if negative balance, "SIPO": stands for household "DUCHOD": stands for old-age pension, "UVER": stands for loan payment
	bank TEXT, -- bank of the partner
	account INTEGER, -- account of the partner
	foreign key (account_id) references account (account_id)
);

-- account.district_id can be joined with district.district_id
-- card.disp_id can be joined with disp.disp_id
-- client.district_id can be joined with district.district_id
-- disp.account_id can be joined with account.account_id
-- disp.client_id can be joined with client.client_id
-- loan.account_id can be joined with account.account_id
-- order.account_id can be joined with account.account_id
-- trans.account_id can be joined with account.account_id

'''


sys_prompt_dc = f'''You are a database administrator and have designed the following database for Debit Card Specializing whose schema is represented as:

CREATE TABLE customers(
	CustomerID INTEGER PRIMARY KEY, -- identification of the customer
	Segment TEXT, -- client segment
	Currency TEXT 
);

CREATE TABLE gasstations(
	GasStationID INTEGER PRIMARY KEY,
	ChainID INTEGER,
	Country TEXT,
	Segment TEXT  
);

CREATE TABLE products(
	ProductID INTEGER PRIMARY KEY,
	Description TEXT  
);

CREATE TABLE yearmonth(
	CustomerID INTEGER PRIMARY KEY,
	Date INTEGER PRIMARY KEY,
	Consumption REAL,
			 
);

CREATE TABLE "transactions_1k"(
	TransactionID INTEGER PRIMARY KEY,
	Date DATE,
	Time TEXT,
	CustomerID INTEGER,
	CardID INTEGER,
	GasStationID INTEGER,
	ProductID INTEGER,
	Amount INTEGER,
	Price REAL
);

-- customers.CustomerID can be joined with yearmonth.CustomerID

'''

sys_prompt_tox = f'''You are a database administrator and have designed the following database for Toxicology whose schema is represented as:

CREATE TABLE atom(
   atom_id TEXT PRIMARY KEY, -- the unique id of atoms
   molecule_id TEXT, -- identifying the molecule to which the atom belongs. TRXXX_i represents ith atom of molecule TRXXX
   element TEXT, -- the element of the toxicology . cl: chlorine, c: carbon, h: hydrogen, o: oxygen, s: sulfur, n: nitrogen, p: phosphorus, na: sodium, br: bromine, f: fluorine, i: iodine, sn: Tin, pb: lead, te: tellurium, ca: Calcium
);

CREATE TABLE bond (
   bond_id TEXT PRIMARY KEY, -- unique id representing bonds. TRxxx_A1_A2:TRXXX refers to which molecule A1 and A2 refers to which atom
   molecule_id TEXT, -- identifying the molecule in which the bond appears
   bond_type  TEXT, -- type of the bond. -: single bond, '=': double bond, '#': triple bond
);

CREATE TABLE connected  (
   atom_id TEXT PRIMARY KEY, -- id of the first atom
   atom_id2 TEXT PRIMARY KEY, -- id of the second atom
   bond_id TEXT, -- bond id representing bond between two atoms
);

CREATE TABLE molecule(
   molecule_id TEXT PRIMARY KEY, -- unique id of molecule. "+" --> this molecule / compound is carcinogenic, '-' this molecule is not / compound carcinogenic
   label TEXT,
);

-- atom.molecule_id can be joined with molecule.molecule_id
-- bond.molecule_id can be joined with molecule.molecule_id
-- connected.atom_id can be joined with atom.atom_id
-- connected.atom_id2 can be joined with atom.atom_id
-- connected.bond_id can be joined with bond.bond_id

'''

sys_prompt_ef = f'''You are a database administrator and have designed the following database for European Football 2 whose schema is represented as:

CREATE TABLE "Player_Attributes" (
	 id 	INTEGER PRIMARY KEY, -- the unique id for players. 
	 player_fifa_api_id 	INTEGER, -- the id of the player fifa api
	 player_api_id 	INTEGER, -- the id of the player api
	 date 	TEXT, -- e.g. 2016-02-18 00:00:00
	 overall_rating 	INTEGER, -- the overall rating of the player. The rating is between 0-100 which is calculated by FIFA. Higher overall rating means the player has a stronger overall strength.
	 potential 	INTEGER, -- potential of the player. The potential score is between 0-100 which is calculated by FIFA. Higher potential score means that the player has more potential
	 preferred_foot 	TEXT, -- the player's preferred foot when attacking. right/ left.
	 attacking_work_rate 	TEXT, -- the player's attacking work rate. high: implies that the player is going to be in all of your attack moves. medium: implies that the player will select the attack actions he will join in. low: remain in his position while the team attacks 
	 defensive_work_rate 	TEXT, -- the player's defensive work rate. high: remain in his position and defense while the team attacks.  medium: implies that the player will select the defensive actions he will join in. low: implies that the player is going to be in all of your attack moves instead of defensing
	 crossing 	INTEGER, -- the player's crossing score. Cross is a long pass into the opponent's goal towards the header of sixth-yard teammate. The crossing score is between 0-100 which measures the tendency/frequency of crosses in the box. 
	 finishing 	INTEGER, -- the player's finishing rate. 0-100 which is calculated by FIFA.
	 heading_accuracy 	INTEGER, -- the player's heading accuracy. 0-100 which is calculated by FIFA.
	 short_passing 	INTEGER, -- the player's short passing score. 0-100 which is calculated by FIFA.
	 volleys 	INTEGER, -- the player's volley score. 0-100 which is calculated by FIFA.
	 dribbling 	INTEGER, -- the player's dribbling score. 0-100 which is calculated by FIFA.
	 curve 	INTEGER, -- the player's curve score. 0-100 which is calculated by FIFA.
	 free_kick_accuracy 	INTEGER, -- the player's free kick accuracy. 0-100 which is calculated by FIFA.
	 long_passing 	INTEGER, -- the player's long passing score. 0-100 which is calculated by FIFA.
	 ball_control 	INTEGER, -- the player's ball control score. 0-100 which is calculated by FIFA
	 acceleration 	INTEGER, -- the player's acceleration score. 0-100 which is calculated by FIFA
	 sprint_speed 	INTEGER, -- the player's sprint speed. 0-100 which is calculated by FIFA
	 agility 	INTEGER, -- the player's agility. 0-100 which is calculated by FIFA
	 reactions 	INTEGER, -- the player's reactions score. 0-100 which is calculated by FIFA
	 balance 	INTEGER, -- the player's balance score. 0-100 which is calculated by FIFA
	 shot_power 	INTEGER, -- the player's shot power. 0-100 which is calculated by FIFA
	 jumping 	INTEGER, -- the player's jumping score. 0-100 which is calculated by FIFA
	 stamina 	INTEGER, -- the player's stamina score. 0-100 which is calculated by FIFA
	 strength 	INTEGER, -- the player's strength score. 0-100 which is calculated by FIFA
	 long_shots 	INTEGER, -- the player's long shots score. 0-100 which is calculated by FIFA
	 aggression 	INTEGER, -- the player's aggression score. 0-100 which is calculated by FIFA
	 interceptions 	INTEGER, -- the player's interceptions score. 0-100 which is calculated by FIFA
	 positioning 	INTEGER, -- the player's positioning score. 0-100 which is calculated by FIFA
	 vision 	INTEGER, -- the player's vision score. 0-100 which is calculated by FIFA
	 penalties 	INTEGER, -- the player's penalties score. 0-100 which is calculated by FIFA. 0-100 which is calculated by FIFA
	 marking 	INTEGER, -- the player's markingscore. 0-100 which is calculated by FIFA. 0-100 which is calculated by FIFA
	 standing_tackle 	INTEGER, -- the player's standing tackle score. 0-100 which is calculated by FIFA
	 sliding_tackle 	INTEGER, -- the player's sliding tackle score. 0-100 which is calculated by FIFA
	 gk_diving 	INTEGER, -- the player's goalkeep diving score. 0-100 which is calculated by FIFA
	 gk_handling 	INTEGER, -- the player's goalkeep diving score. 0-100 which is calculated by FIFA
	 gk_kicking 	INTEGER, -- the player's goalkeep kicking score. 0-100 which is calculated by FIFA
	 gk_positioning 	INTEGER, -- the player's goalkeep positioning score. 0-100 which is calculated by FIFA
	 gk_reflexes 	INTEGER -- the player's goalkeep reflexes score. 0-100 which is calculated by FIFA
);

CREATE TABLE  Player  (
	 id 	INTEGER PRIMARY KEY, -- the unique id for players
	 player_api_id 	INTEGER, -- the id of the player api
	 player_name 	TEXT, -- player name
	 player_fifa_api_id 	INTEGER, -- the id of the player fifa api
	 birthday 	TEXT, -- the player's birthday. e.g. 1992-02-29 00:00:00. Player A is older than player B means that A's birthday is earlier than B's
	 height 	INTEGER, -- the player's height
	 weight 	INTEGER -- the player's weight
);

CREATE TABLE  Match  (
	 id 	INTEGER PRIMARY KEY, -- the unique id for matches
	 country_id 	INTEGER, -- country id
	 league_id 	INTEGER, -- league id
	 season 	TEXT, -- the season of the match
	 stage 	INTEGER, -- the stage of the match
	 date 	TEXT, -- the date of the match. e.g. 2008-08-17 00:00:00
	 match_api_id 	INTEGER  , -- the id of the match api
	 home_team_api_id 	INTEGER, -- the id of the home team api
	 away_team_api_id 	INTEGER, -- the id of the away team api
	 home_team_goal 	INTEGER, -- the goal of the home team
	 away_team_goal 	INTEGER, -- the goal of the away team
	 home_player_X1 	INTEGER,
	 home_player_X2 	INTEGER,
	 home_player_X3 	INTEGER,
	 home_player_X4 	INTEGER,
	 home_player_X5 	INTEGER,
	 home_player_X6 	INTEGER,
	 home_player_X7 	INTEGER,
	 home_player_X8 	INTEGER,
	 home_player_X9 	INTEGER,
	 home_player_X10 	INTEGER,
	 home_player_X11 	INTEGER,
	 away_player_X1 	INTEGER,
	 away_player_X2 	INTEGER,
	 away_player_X3 	INTEGER,
	 away_player_X4 	INTEGER,
	 away_player_X5 	INTEGER,
	 away_player_X6 	INTEGER,
	 away_player_X7 	INTEGER,
	 away_player_X8 	INTEGER,
	 away_player_X9 	INTEGER,
	 away_player_X10 	INTEGER,
	 away_player_X11 	INTEGER,
	 home_player_Y1 	INTEGER,
	 home_player_Y2 	INTEGER,
	 home_player_Y3 	INTEGER,
	 home_player_Y4 	INTEGER,
	 home_player_Y5 	INTEGER,
	 home_player_Y6 	INTEGER,
	 home_player_Y7 	INTEGER,
	 home_player_Y8 	INTEGER,
	 home_player_Y9 	INTEGER,
	 home_player_Y10 	INTEGER,
	 home_player_Y11 	INTEGER,
	 away_player_Y1 	INTEGER,
	 away_player_Y2 	INTEGER,
	 away_player_Y3 	INTEGER,
	 away_player_Y4 	INTEGER,
	 away_player_Y5 	INTEGER,
	 away_player_Y6 	INTEGER,
	 away_player_Y7 	INTEGER,
	 away_player_Y8 	INTEGER,
	 away_player_Y9 	INTEGER,
	 away_player_Y10 	INTEGER,
	 away_player_Y11 	INTEGER,
	 home_player_1 	INTEGER,
	 home_player_2 	INTEGER,
	 home_player_3 	INTEGER,
	 home_player_4 	INTEGER,
	 home_player_5 	INTEGER,
	 home_player_6 	INTEGER,
	 home_player_7 	INTEGER,
	 home_player_8 	INTEGER,
	 home_player_9 	INTEGER,
	 home_player_10 	INTEGER,
	 home_player_11 	INTEGER,
	 away_player_1 	INTEGER,
	 away_player_2 	INTEGER,
	 away_player_3 	INTEGER,
	 away_player_4 	INTEGER,
	 away_player_5 	INTEGER,
	 away_player_6 	INTEGER,
	 away_player_7 	INTEGER,
	 away_player_8 	INTEGER,
	 away_player_9 	INTEGER,
	 away_player_10 	INTEGER,
	 away_player_11 	INTEGER,
	 goal 	TEXT, -- the goal of the match
	 shoton 	TEXT, -- the shot on goal of the match
	 shotoff 	TEXT, -- the shot off goal of the match, which is the opposite of shot on
	 foulcommit 	TEXT, -- the fouls occurred in the match
	 card 	TEXT, -- the cards given in the match
	 cross 	TEXT, -- Balls sent into the opposition team's area from a wide position in the match
	 corner 	TEXT, -- Ball goes out of play for a corner kick in the match
	 possession 	TEXT, -- The duration from a player taking over the ball in the match
	 B365H 	NUMERIC,
	 B365D 	NUMERIC,
	 B365A 	NUMERIC,
	 BWH 	NUMERIC,
	 BWD 	NUMERIC,
	 BWA 	NUMERIC,
	 IWH 	NUMERIC,
	 IWD 	NUMERIC,
	 IWA 	NUMERIC,
	 LBH 	NUMERIC,
	 LBD 	NUMERIC,
	 LBA 	NUMERIC,
	 PSH 	NUMERIC,
	 PSD 	NUMERIC,
	 PSA 	NUMERIC,
	 WHH 	NUMERIC,
	 WHD 	NUMERIC,
	 WHA 	NUMERIC,
	 SJH 	NUMERIC,
	 SJD 	NUMERIC,
	 SJA 	NUMERIC,
	 VCH 	NUMERIC,
	 VCD 	NUMERIC,
	 VCA 	NUMERIC,
	 GBH 	NUMERIC,
	 GBD 	NUMERIC,
	 GBA 	NUMERIC,
	 BSH 	NUMERIC,
	 BSD 	NUMERIC,
	 BSA 	NUMERIC
);

CREATE TABLE  League  (
	 id 	INTEGER PRIMARY KEY, -- the unique id for leagues
	 country_id 	INTEGER, -- the unique id for countries
	 name 	TEXT -- league name
);

CREATE TABLE  Country  (
	 id 	INTEGER PRIMARY KEY, -- the unique id for countries
	 name 	TEXT -- country name
);

CREATE TABLE "Team" (
	 id 	INTEGER PRIMARY KEY, -- the unique id for teams
	 team_api_id 	INTEGER  , -- the id of the team api
	 team_fifa_api_id 	INTEGER, -- the id of the team fifa api
	 team_long_name 	TEXT, -- the team's long name
	 team_short_name 	TEXT -- the team's short name
);

CREATE TABLE  Team_Attributes  (
	 id 	INTEGER PRIMARY KEY, -- the unique id for teams
	 team_fifa_api_id 	INTEGER, -- the id of the team fifa api
	 team_api_id 	INTEGER, -- the id of the team api
	 date 	TEXT, -- Date. e.g. 2010-02-22 00:00:00
	 buildUpPlaySpeed 	INTEGER, -- the speed in which attacks are put together. the score which is between 1-00 to measure the team's attack speed
	 buildUpPlaySpeedClass 	TEXT, -- the speed class. Slow: 1-33, Balanced: 34-66, Fast: 66-100
	 buildUpPlayDribbling 	INTEGER, -- the tendency/ frequency of dribbling
	 buildUpPlayDribblingClass 	TEXT, -- the dribbling class. Little: 1-33. Normal: 34-66. Lots: 66-100
	 buildUpPlayPassing 	INTEGER, -- affects passing distance and support from teammates
	 buildUpPlayPassingClass 	TEXT, -- the passing class. Short: 1-33. Mixed: 34-66. Long: 66-100
	 buildUpPlayPositioningClass 	TEXT, -- A team's freedom of movement in the 1st two thirds of the pitch. Organised / Free Form
	 chanceCreationPassing 	INTEGER, -- Amount of risk in pass decision and run support
	 chanceCreationPassingClass 	TEXT, -- the chance creation passing class. Safe: 1-33. Normal: 34-66. Risky: 66-100.
	 chanceCreationCrossing 	INTEGER, -- The tendency / frequency of crosses into the box
	 chanceCreationCrossingClass 	TEXT, -- the chance creation crossing class. Little: 1-33. Normal: 34-66. Lots: 66-100.
	 chanceCreationShooting 	INTEGER, -- The tendency / frequency of shots taken
	 chanceCreationShootingClass 	TEXT, -- the chance creation shooting class. Little: 1-33. Normal: 34-66. Lots: 66-100.
	 chanceCreationPositioningClass 	TEXT, -- A team�s freedom of movement in the final third of the pitch. Organised / Free Form
	 defencePressure 	INTEGER, -- Affects how high up the pitch the team will start pressuring
	 defencePressureClass 	TEXT, -- the defence pressure class. Deep: 1-33. Medium: 34-66. High: 66-100.
	 defenceAggression 	INTEGER, -- Affect the team�s approach to tackling the ball possessor
	 defenceAggressionClass 	TEXT, -- the defence aggression class. Contain: 1-33. Press: 34-66. Double: 66-100.
	 defenceTeamWidth 	INTEGER, -- Affects how much the team will shift to the ball side
	 defenceTeamWidthClass 	TEXT, -- the defence team width class. Narrow: 1-33. Normal: 34-66. Wide: 66-100.
	 defenceDefenderLineClass 	TEXT -- Affects the shape and strategy of the defence. Cover/ Offside Trap.
)


-- Player_Attributes.player_fifa_api_id can be joined with Player.player_fifa_api_id
-- Player_Attributes.player_api_id can be joined with Player.player_api_id
-- Match.country_id can be joined with country.id
-- Match.league_id can be joined with Team ( team_api_id )
-- Match.home_team_api_id can be joined with Team.team_api_id
-- Match.home_player_1 can be joined with Player.player_api_id
-- Match.home_player_2 can be joined with Player.player_api_id
-- Match.home_player_3 can be joined with Player.player_api_id
-- Match.home_player_4 can be joined with Player.player_api_id
-- Match.home_player_5 can be joined with Player.player_api_id
-- Match.home_player_6 can be joined with Player.player_api_id
-- Match.home_player_7 can be joined with Player.player_api_id
-- Match.home_player_8 can be joined with Player.player_api_id
-- Match.home_player_9 can be joined with Player.player_api_id
-- Match.home_player_10 can be joined with Player.player_api_id
-- Match.home_player_11 can be joined with Player.player_api_id
-- Match.away_player_1 can be joined with Player.player_api_id
-- Match.away_player_2 can be joined with Player.player_api_id
-- Match.away_player_3 can be joined with Player.player_api_id
-- Match.away_player_4 can be joined with Player.player_api_id
-- Match.away_player_5 can be joined with Player.player_api_id
-- Match.away_player_6 can be joined with Player.player_api_id
-- Match.away_player_7 can be joined with Player.player_api_id
-- Match.away_player_8 can be joined with Player.player_api_id
-- Match.away_player_9 can be joined with Player.player_api_id
-- Match.away_player_10 can be joined with Player.player_api_id
-- Match.away_player_11 can be joined with Player.player_api_id
-- Team_Attributes.team_fifa_api_id can be joined with Team.team_fifa_api_id
-- Team_Attributes.team_api_id can be joined with Team.team_api_id

'''

sys_prompt_sh = f'''You are a database administrator and have designed the following database for Superhero whose schema is represented as:

CREATE TABLE alignment (
    id INTEGER primary key,
    alignment TEXT
)

CREATE TABLE attribute (
    id INTEGER primary key,
    attribute_name TEXT
)

CREATE TABLE colour (
    id INTEGER primary key,
    colour TEXT
)

CREATE TABLE gender (
    id INTEGER primary key,
    gender TEXT
)

CREATE TABLE publisher (
    id INTEGER primary key,
    publisher_name TEXT
)

CREATE TABLE race (
    id INTEGER primary key,
    race TEXT
)

CREATE TABLE superhero (
    id INTEGER primary key,
    superhero_name TEXT,
    full_name TEXT,
    gender_id INTEGER,
    eye_colour_id INTEGER,
    hair_colour_id INTEGER,
    skin_colour_id INTEGER,
    race_id INTEGER,
    publisher_id INTEGER,
    alignment_id INTEGER,
    height_cm INTEGER,
    weight_kg INTEGER
)

CREATE TABLE hero_attribute (
    hero_id  INTEGER,
    attribute_id    INTEGER,
    attribute_value INTEGER
)

CREATE TABLE superpower (
    id  INTEGER primary key,
    power_name TEXT
)

CREATE TABLE hero_power (
    hero_id  INTEGER,
    power_id INTEGER,
    foreign key (hero_id) references superhero(id),
    foreign key (power_id) references superpower(id)
)

-- foreign key superhero.alignment_id references alignment.id,
-- foreign key superhero.eye_colour_id references colour.id,
-- foreign key superhero.gender_id references gender.id,
-- foreign key superhero.hair_colour_id references colour.id,
-- foreign key superhero.publisher_id references publisher.id,
-- foreign key superhero.race_id references race.id,
-- foreign key superhero.skin_colour_id references colour.id
-- foreign key hero_attribute.attribute_id references attribute.id,
-- foreign key hero_attribute.hero_id references superhero.id
-- foreign key hero_power.hero_id references superhero.id,
-- foreign key hero_power.power_id references superpower.id


'''

sys_prompt_sc = f'''You are a database administrator and have designed the following database for Student Club whose schema is represented as:

CREATE TABLE event (
    event_id   TEXT primary key, -- A unique identifier for the event
    event_name TEXT, -- event name
    event_date TEXT, -- The date the event took place or is scheduled to take place 
    type       TEXT, -- The kind of event, such as game, social, election
     es      TEXT, -- A free text field for any notes about the event
    location   TEXT, -- Address where the event was held or is to be held or the name of such a location
    status     TEXT -- One of three values indicating if the event is in planning, is opened, or is closed
)

CREATE TABLE major (
    major_id   TEXT primary key, -- A unique identifier for each major
    major_name TEXT, -- major name
    department TEXT, -- The name of the department that offers the major
    college    TEXT -- The name college that houses the department that offers the major
)

CREATE TABLE zip_code (
    zip_code    INTEGER primary key, -- The ZIP code itself. A five-digit number identifying a US post office. Standard: the normal codes with which most people are familiar, PO Box: zip codes have post office boxes, Unique: zip codes that are assigned to individual organizations. 
    type        TEXT, -- The kind of ZIP code
    city        TEXT, -- The city to which the ZIP pertains
    county      TEXT, -- The county to which the ZIP pertains
    state       TEXT, -- The name of the state to which the ZIP pertains
    short_state TEXT -- The abbreviation of the state to which the ZIP pertains
)

CREATE TABLE "attendance" (
    link_to_event  TEXT primary key, -- The unique identifier of the event which was attended
    link_to_member TEXT primary key, -- The unique identifier of the member who attended the event

)

CREATE TABLE "budget" (
    budget_id     TEXT primary key, -- A unique identifier for the budget entry
    category      TEXT, -- The area for which the amount is budgeted, such as, advertisement, food, parking
    spent         FLOAT, -- The total amount spent in the budgeted category for an event. the unit is dollar. This is summarized from the Expense table
    remaining     FLOAT, -- A value calculated as the amount budgeted minus the amount spent. the unit is dollar, If the remaining < 0, it means that the cost has exceeded the budget.
    amount        INTEGER, -- The amount budgeted for the specified category and event. the unit is dollar, some computation like: amount = spent + remaining  
    event_status  TEXT, -- the status of the event. Closed / Open/ Planning, Closed: It means that the event is closed. The spent and the remaining won't change anymore, Open: It means that the event is already opened. The spent and the remaining will change with new expenses, Planning: The event is not started yet but is planning. The spent and the remaining won't change at this stage. 
    link_to_event TEXT -- The unique identifier of the event to which the budget line applies.
)

CREATE TABLE "expense" (
    expense_id          TEXT primary key, -- unique id of income
    expense_description TEXT, -- A textual description of what the money was spend for
    expense_date        TEXT, -- The date the expense was incurred. e.g. YYYY-MM-DD
    cost                REAL, -- The dollar amount of the expense. the unit is dollar
    approved            TEXT, -- A true or false value indicating if the expense was approved. true/ false
    link_to_member      TEXT, -- The member who incurred the expense
    link_to_budget      TEXT -- The unique identifier of the record in the Budget table that indicates the expected total expenditure for a given category and event. 
)

CREATE TABLE "income" (
    income_id TEXT primary key, -- A unique identifier for each record of income
    date_received  TEXT, -- the date that the fund received
    amount INTEGER, -- amount of funds. the unit is dollar
    source TEXT, -- A value indicating where the funds come from such as dues, or the annual university allocation
     es TEXT, -- A free-text value giving any needed details about the receipt of funds
    link_to_member TEXT -- link to member
)

CREATE TABLE "member" (
    member_id TEXT primary key, -- unique id of member
    first_name    TEXT, -- member's first name
    last_name     TEXT, -- member's last name. full name is first_name + last_name. e.g. A member's first name is Angela and last name is Sanders. Thus, his/her full name is Angela Sanders.
    email         TEXT, -- member's email
    position      TEXT, -- The position the member holds in the club
    t_shirt_size  TEXT, -- The size of tee shirt that member wants when shirts are ordered. usually the student ordered t-shirt with lager size has bigger body shape
    phone         TEXT, -- The best telephone at which to contact the member
    zip           INTEGER, -- the zip code of the member's hometown
    link_to_major TEXT -- The unique identifier of the major of the member. References the Major table
)

-- attendance.link_to_event can be joined with event.event_id
-- attendance.link_to_member can be joined with member.member_id
-- budget.link_to_event can be joined with event.event_id
-- expense.link_to_member can be joined with member.member_id
-- expense.link_to_budget can be joined with budget.budget_id
-- income.link_to_member can be joined with member.member_id
-- member.zip can be joined with zip_code.zip_code
-- member.link_to_major can be joined with major.major_id

'''

sys_prompt_tp = '''You are a database administrator and have designed the following database for Thrombosis Prediction whose schema is represented as:

CREATE TABLE Examination (
  ID INTEGER PRIMARY KEY, -- identification of the patient
  `Examination Date` DATE, --Format Year-Month-Date;
  `aCL IgG` REAL, -- anti-Cardiolipin antibody (IgG) concentration
  `aCL IgM` REAL, -- anti-Cardiolipin antibody (IgM) concentration
  ANA INTEGER , -- anti-nucleus antibody concentration
  `ANA Pattern` TEXT , --pattern observed in the sheet of ANA examination
  `aCL IgA` INTEGER ,  -- anti-Cardiolipin antibody (IgA) concentration
  Diagnosis TEXT ,  -- disease names
  KCT TEXT , -- measure of degree of coagulation
  RVVT TEXT ,  --measure of degree of coagulation;
  LAC TEXT ,  --measure of degree of coagulation
  Symptoms TEXT , -- other symptoms observed;
  Thrombosis INTEGER , --degree of thrombosis
);
 
CREATE TABLE Patient ( 
  ID INTEGER  PRIMARY KEY,  -- identification of the patient
  SEX TEXT , -- Sex 
  Birthday DATE, --Format year-Month-Date;
  Description DATE , -- the first date when a patient data was recorded;
  `First Date` DATE , -- the date when a patient came to the hospital;
  Admission TEXT , -- patient was admitted to the hospital (+) or followed at the outpatient clinic (-);
  Diagnosis TEXT  -- disease names;
);

CREATE TABLE Laboratory (
  ID INTEGER PRIMARY KEY, -- identification of the patient;
  Date DATE PRIMARY KEY, -- Date of the laboratory tests (YYMMDD); --Format Year-Month-Date;
  GOT INTEGER , -- AST glutamic oxaloacetic transaminase. Normal range: N < 60
  GPT INTEGER , -- ALT glutamic pyruvic transaminase. Normal range: N < 60;
  LDH INTEGER , -- lactate dehydrogenase. Normal range: N < 500;
  ALP INTEGER , --  alkaliphophatase. Normal range: N < 300
  TP REAL , -- total protein. Normal range: 6.0 < N < 8.5;
  ALB REAL , -- albumin. Normal range: 3.5 < N < 5.5;
  UA REAL , -- uric acid. Normal range: N > 8.0 (Male)N > 6.5 (Female);
  UN INTEGER , -- urea nitrogen. Normal range: N < 30;
  CRE REAL , -- creatinine. Normal range: N < 1.5;
  `T-BIL` REAL , -- total bilirubin. Normal range: N < 2.0
  `T-CHO` INTEGER , -- total cholesterol. Normal range: N < 250;
  TG INTEGER , -- triglyceride. Normal range: N < 200
  CPK INTEGER , -- creatinine phosphokinase. Normal range: N < 250
  GLU INTEGER , -- blood glucose. Normal range: N < 180
  WBC REAL , -- White blood cell. Normal range: 3.5 < N < 9.0;
  RBC REAL , -- Red blood cell. Normal range: 3.5 < N < 6.0
  HGB REAL , -- Hemoglobin. Normal range: 10 < N < 17
  HCT REAL , -- Hematoclit. Normal range: 29 < N < 52
  PLT INTEGER ,--  platelet. Normal range: 100 < N < 400
  PT REAL , -- prothrombin time. Normal range: N < 14
  APTT INTEGER , -- activated partial prothrombin time. Normal range: N < 45
  FG REAL , -- fibrinogen. Normal range: 150 < N < 450
  PIC INTEGER ,
  TAT INTEGER ,
  TAT2 INTEGER ,
  `U-PRO` TEXT , -- proteinuria Normal range: 0 < N < 30
  IGG INTEGER , -- Ig G. Normal range: 900 < N < 2000
  IGA INTEGER , -- Ig A. Normal range: 80 < N < 500
  IGM INTEGER , -- Ig M. Normal range: 40 < N < 400
  CRP TEXT , -- C-reactive protein. Normal range: N= -, +-, or N < 1.0;
  RA TEXT , -- RAHA. Normal range: N= -, +-
  RF TEXT , -- Rhuematoid Factor. Normal range: N < 20
  C3 INTEGER ,--  complement 3. Normal range: N > 35
  C4 INTEGER , -- complement 4. Normal range: N > 10
  RNP TEXT , -- anti-ribonuclear protein. Normal range: N= -, +-
  SM TEXT , -- anti-SM. Normal range: N= -, +-
  SC170 TEXT ,--  anti-scl70. Normal range: N= -, +-
  SSA TEXT , -- anti-SSA . Normal range: N= -, +-
  SSB TEXT , -- anti-SSB. Normal range: N= -, +-
  CENTROMEA TEXT ,--  anti-centromere. Normal range: N= -, +-
  DNA TEXT , -- anti-DNA. Normal range: N < 8
  DNA-II INTEGER ,-- anti-DNA. Normal range: N < 8
);
'''

sys_prompt_cs = '''You are a database administrator and have designed the following database for California Schools whose schema is represented as:

CREATE TABLE frpm (
  CDSCode INTEGER PRIMARY KEY,
  `Academic Year` INTEGER,
  `County Code` INTEGER,
  `District Code` INTEGER,
  `School Code` INTEGER,
  `County Name` TEXT,
  `District Name` TEXT,
  `School Name` TEXT,
  `District Type` TEXT,
  `School Type` TEXT,
  `Educational Option Type` TEXT,
  `NSLP Provision Status` TEXT,
  `Charter School (Y/N)` INTEGER,
  `Charter School Number` TEXT,
  `Charter Funding Type` TEXT,
  IRC INTEGER,
  `Low Grade` TEXT,
  `High Grade` TEXT,
  `Enrollment (K-12)` REAL,
  `Free Meal Count (K-12)` REAL,
  `FRPM Count (K-12)` REAL,
  `Enrollment (Ages 5-17)` REAL,
  `Free Meal Count (Ages 5-17)` REAL,
  `Percent (%) Eligible Free (K-12)` REAL,
  `Percent (%) Eligible Free (Ages 5-17)` REAL,
  `Percent (%) Eligible FRPM (Ages 5-17)` REAL,
  `2013-14 CALPADS Fall 1 Certification Status` REAL
);

CREATE TABLE satscores (
  cds TEXT PRIMARY KEY,
  rtype TEXT,
  sname TEXT,
  dname TEXT,
  cname TEXT,
  enroll12 INTEGER,
  NumTstTakr INTEGER,
  AvgScrRead INTEGER,
  AvgScrMath INTEGER,
  AvgScrWrite INTEGER,
  NumGE1500 INTEGER
);

CREATE TABLE schools (
  CDSCode TEXT PRIMARY KEY,
  NCESDist TEXT,
  NCESSchool TEXT,
  StatusType TEXT,
  County TEXT,
  District TEXT,
  School TEXT,
  Street TEXT,
  StreetAbr TEXT,
  City TEXT,
  Zip TEXT,
  State TEXT,
  MailStreet TEXT,
  MailStrAbr TEXT,
  MailCity TEXT,
  MailZip TEXT,
  MailState TEXT,
  Phone TEXT,
  Ext TEXT,
  Website TEXT,
  OpenDate DATE,
  ClosedDate DATE,
  Charter INTEGER,
  CharterNum TEXT,
  FundingType TEXT,
  DOC TEXT,
  DOCType TEXT,
  SOC TEXT,
  SOCType TEXT,
  EdOpsCode TEXT,
  EdOpsName TEXT,
  EILCode TEXT,
  EILName TEXT,
  GSoffered TEXT,
  GSserved TEXT,
  Virtual TEXT,
  Magnet TEXT,
  Latitude REAL,
  Longitude REAL,
  AdmFName1 TEXT,
  AdmLName1 TEXT,
  AdmEmail1 TEXT,
  AdmFName2 TEXT,
  AdmLName2 TEXT,
  AdmEmail2 TEXT,
  AdmFName3 TEXT,
  AdmLName3 TEXT,
  AdmEmail3 TEXT,
  LastUpdate DATE
);
'''

sys_prompt_f1 = '''You are a database administrator and have designed the following database for Formula 1 whose schema is represented as:

CREATE TABLE circuits (
 `circuitId` INTEGER PRIMARY KEY,
 `circuitRef` TEXT,
 `name` TEXT,
 `location` TEXT,
 `country` TEXT,
 `lat` REAL,
 `lng` REAL,
 `alt` INTEGER,
 `url` TEXT
);

CREATE TABLE constructors (
 `constructorId` INTEGER PRIMARY KEY, 
 `constructorRef` TEXT, 
 `name` TEXT, 
 `nationality` TEXT, 
 `url` TEXT
);

CREATE TABLE drivers (
 `driverId` INTEGER PRIMARY KEY,
 `driverRef` TEXT,
 `number` INTEGER,
 `code` TEXT,
 `forename` TEXT,
 `surname` TEXT,
 `dob` DATE,
 `nationality` TEXT,
 `url` TEXT
);

CREATE TABLE seasons (
  `year` INTEGER PRIMARY KEY,
  `url` TEXT
);

CREATE TABLE races (
 `raceId` INTEGER PRIMARY KEY,
 `year` INTEGER, 
 `round` INTEGER, 
 `circuitId` INTEGER, 
 `name` TEXT, 
 `date` DATE, 
 `time` TEXT, 
 `url` TEXT
);

CREATE TABLE constructorResults (
 `constructorResultsId` INTEGER PRIMARY KEY,
 `raceId` INTEGER, 
 `constructorId` INTEGER, 
 `points` REAL, 
 `status` TEXT
);

CREATE TABLE constructorStandings (
 `constructorStandingsId` INTEGER PRIMARY KEY,
 `raceId` INTEGER,
 `constructorId` INTEGER,
 `points` INTEGER,
 `position` INTEGER,
 `positionText` TEXT,
 `wins` INTEGER
);

CREATE TABLE driverStandings (
 `driverStandingsId` INTEGER PRIMARY KEY,
 `raceId` INTEGER,
 `driverId` INTEGER,
 `points` REAL,
 `position` INTEGER,
 `positionText` TEXT,
 `wins` INTEGER
);

CREATE TABLE lapTimes (
 `raceId` INTEGER PRIMARY KEY, 
 `driverId` INTEGER, 
 `lap` INTEGER, 
 `position` INTEGER, 
 `time` TEXT, 
 `milliseconds` INTEGER
);

CREATE TABLE pitStops (
 `raceId` INTEGER PRIMARY KEY,  
 `driverId` INTEGER, 
 `stop` INTEGER, 
 `lap` INTEGER, 
 `time` TEXT, 
 `duration` TEXT, 
 `milliseconds` INTEGER
);

CREATE TABLE qualifying (
 `qualifyId` INTEGER PRIMARY KEY,
 `raceId` INTEGER,
 `driverId` INTEGER,
 `constructorId` INTEGER,
 `number` INTEGER,
 `position` INTEGER,
 `q1` TEXT,
 `q2` TEXT,
 `q3` TEXT
);

CREATE TABLE status (
 `statusId` INTEGER PRIMARY KEY, 
 `status` TEXT
);

CREATE TABLE results (
 `resultId` INTEGER PRIMARY KEY, 
 `raceId` INTEGER,
 `driverId` INTEGER,
 `constructorId` INTEGER,
 `number` INTEGER,
 `grid` INTEGER,
 `position` INTEGER,
 `positionText` TEXT,
 `positionOrder` INTEGER,
 `points` REAL,
 `laps` INTEGER,
 `time` TEXT,
 `milliseconds` INTEGER,
 `fastestLap` INTEGER,
 `rank` INTEGER,
 `fastestLapTime` TEXT,
 `fastestLapSpeed` TEXT,
 `statusId` INTEGER
);
'''

sys_prompt_cg = '''You are a database administrator and have designed the following database for Card Games whose schema is represented as:

CREATE TABLE cards (
 `id` INTEGER PRIMARY KEY,
 `artist` TEXT,
 `asciiName` TEXT,
 `availability` TEXT,
 `borderColor` TEXT,
 `cardKingdomFoilId` TEXT,
 `cardKingdomId` TEXT,
 `colorIdentity` TEXT,
 `colorIndicator` TEXT,
 `colors` TEXT,
 `convertedManaCost` REAL,
 `duelDeck` TEXT,
 `edhrecRank` INTEGER,
 `faceConvertedManaCost` REAL,
 `faceName` TEXT,
 `flavorName` TEXT,
 `flavorText` TEXT,
 `frameEffects` TEXT,
 `frameVersion` TEXT,
 `hand` TEXT,
 `hasAlternativeDeckLimit` INTEGER,
 `hasContentWarning` INTEGER,
 `hasFoil` INTEGER,
 `hasNonFoil` INTEGER,
 `isAlternative` INTEGER,
 `isFullArt` INTEGER,
 `isOnlineOnly` INTEGER,
 `isOversized` INTEGER,
 `isPromo` INTEGER,
 `isReprint` INTEGER,
 `isReserved` INTEGER,
 `isStarter` INTEGER,
 `isStorySpotlight` INTEGER,
 `isTextless` INTEGER,
 `isTimeshifted` INTEGER,
 `keywords` TEXT,
 `layout` TEXT,
 `leadershipSkills` TEXT,
 `life` TEXT,
 `loyalty` TEXT,
 `manaCost` TEXT,
 `mcmId` TEXT,
 `mcmMetaId` TEXT,
 `mtgArenaId` TEXT,
 `mtgjsonV4Id` TEXT,
 `mtgoFoilId` TEXT,
 `mtgoId` TEXT,
 `multiverseId` TEXT,
 `name` TEXT,
 `number` TEXT,
 `originalReleaseDate` TEXT,
 `originalText` TEXT,
 `originalType` TEXT,
 `otherFaceIds` TEXT,
 `power` TEXT,
 `printings` TEXT,
 `promoTypes` TEXT,
 `purchaseUrls` TEXT,
 `rarity` TEXT,
 `scryfallId` TEXT,
 `scryfallIllustrationId` TEXT,
 `scryfallOracleId` TEXT,
 `setCode` TEXT,
 `side` TEXT,
 `subtypes` TEXT,
 `supertypes` TEXT,
 `tcgplayerProductId` TEXT,
 `text` TEXT,
 `toughness` TEXT,
 `type` TEXT,
 `types` TEXT,
 `uuid` TEXT,
 `variations` TEXT,
 `watermark` TEXT
);

CREATE TABLE foreign_data (
 `id` INTEGER PRIMARY KEY,
 `flavorText` TEXT,
 `language` TEXT,
 `multiverseid` INTEGER,
 `name` TEXT,
 `text` TEXT,
 `type` TEXT,
 `uuid` TEXT
);

CREATE TABLE legalities (
 `id` INTEGER PRIMARY KEY,
`format` TEXT,
 `status` TEXT,
 `uuid` TEXT
);

CREATE TABLE rulings (
  `id` INTEGER PRIMARY KEY, 
  `date` DATE, 
  `text` TEXT, 
  `uuid` TEXT
);

CREATE TABLE Sets (
 `id` INTEGER PRIMARY KEY,
 `baseSetSize` INTEGER,
 `block` TEXT,
 `booster` TEXT,
 `code` TEXT,
 `isFoilOnly` INTEGER,
 `isForeignOnly` INTEGER,
 `isNonFoilOnly` INTEGER,
 `isOnlineOnly` INTEGER,
 `isPartialPreview` INTEGER,
 `keyruneCode` TEXT,
 `mcmId` INTEGER,
 `mcmIdExtras` INTEGER,
 `mcmName` TEXT,
 `mtgoCode` TEXT,
 `name` TEXT,
 `parentCode` TEXT,
 `releaseDate` DATE,
 `tcgplayerGroupId` INTEGER,
 `totalSetSize` INTEGER,
 `type` TEXT
);

CREATE TABLE Set_translations (
  `id` INTEGER PRIMARY KEY, 
  `language` TEXT, 
  `setCode` TEXT, 
  `translation` TEXT
);
'''

sys_prompt_cc = '''You are a database administrator and have designed the following database for Codebase Community whose schema is represented as:

CREATE TABLE badges (
 `Id` INTEGER PRIMARY KEY,
 `UserId` INTEGER, 
 `Name` TEXT, 
 `Date` DATETIME
);

CREATE TABLE comments (
 `Id` INTEGER PRIMARY KEY, 
 `PostId` INTEGER, 
 `Score` INTEGER, 
 `Text` TEXT, 
 `CreationDate` DATETIME, 
 `UserId` INTEGER, 
 `UserDisplayName` TEXT
);

CREATE TABLE postHistory (
 `Id` INTEGER PRIMARY KEY,
 `PostHistoryTypeId` INTEGER,
 `PostId` INTEGER,
 `RevisionGUID` INTEGER,
 `CreationDate` DATETIME,
 `UserId` INTEGER,
 `Text` TEXT,
 `Comment` TEXT,
 `UserDisplayName` TEXT
);

CREATE TABLE postLinks (
  `Id` INTEGER PRIMARY KEY, 
  `CreationDate` DATETIME, 
  `PostId` INTEGER, 
  `RelatedPostId` INTEGER, 
  `LinkTypeId` INTEGER
);

CREATE TABLE posts (
 `Id` INTEGER PRIMARY KEY, 
 `PostTypeId` INTEGER,
 `AcceptedAnswerId` INTEGER,
 `CreaionDate` DATETIME,
 `Score` INTEGER,
 `ViewCount` INTEGER,
 `Body` TEXT,
 `OwnerUserId` INTEGER,
 `LasActivityDate` DATETIME,
 `Title` TEXT,
 `Tags` TEXT,
 `AnswerCount` INTEGER,
 `CommentCount` INTEGER,
 `FavoriteCount` INTEGER,
 `LastEditorUserId` INTEGER,
 `LastEditDate` DATETIME,
 `CommunityOwnedDate` DATETIME,
 `ParentId` INTEGER,
 `ClosedDate` DATEFORMAT,
 `OwnerDisplayName` TEXT,
 `LastEditorDisplayName` TEXT
);

CREATE TABLE tags (
  `Id` INTEGER PRIMARY KEY,  
  `TagName` TEXT, 
  `Count` INTEGER, 
  `ExcerptPostId` INTEGER, 
  `WikiPostId` TEXT
);

CREATE TABLE users (
  `Id` INTEGER PRIMARY KEY,  
 `Reputation` INTEGER,
 `CreationDate` DATETIME,
 `DisplayName` TEXT,
 `LastAccessDate` DATETIME,
 `WebsiteUrl` TEXT,
 `Location` TEXT,
 `AboutMe` TEXT,
 `Views` INTEGER,
 `UpVotes` INTEGER,
 `DownVotes` INTEGER,
 `AccountId` INTEGER,
 `Age` INTEGER,
 `ProfileImageUrl` TEXT
);

CREATE TABLE votes (
  `Id` INTEGER PRIMARY KEY,
  `PostId` INTEGER, 
  `VoteTypeId` INTEGER, 
  `CreationDate` DATETIME, 
  `UserId` INTEGER, 
  `BountyAmount` INTEGER
);
'''


folder_2 = 'outputs_bm25_diff_recall'

# Define the function to get an answer from OpenAI's API
def get_answer(sys_prompt, prompt, n, temperature=0, answer=[]):
    # Prepare messages based on whether there are previous answers
    if len(answer) > 0:            
        messages = [{"role": "system", "content": sys_prompt}]
        for i in range(len(answer)):
            messages.append({"role": "user", "content": prompt[i]})
            messages.append({"role": "assistant", "content": answer[i]})
        messages.append({"role": "user", "content": prompt[-1]})
    else:
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
        ]
 
    try:
        # Make the API call to get a response from OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            temperature=temperature,
            messages=messages
        )
    except openai.error.RateLimitError as e:
        # Handle rate limit error by retrying after a delay
        print(f"Exception: {e} occurred at Sample: {n}")
        print(f"Waiting for 2 Seconds....")
        for _ in tqdm(range(2), desc="Second(s) Passed"):
            time.sleep(1)
        print(f"Continuing at Sample: {n}....")
        response = get_answer(sys_prompt, prompt, n, temperature, answer)
    except openai.error.InvalidRequestError as e:
        # Handle invalid request error by retrying after a delay
        print(f"Exception: {e} occurred at Sample: {n}")
        print(f"Waiting for 2 Seconds....")
        for _ in tqdm(range(2), desc="Second(s) Passed"):
            time.sleep(1)
        print(f"Continuing at Sample: {n}....")
        response = get_answer("", prompt, n, temperature, answer)
    except Exception as e:
        # Handle other exceptions by retrying after a longer delay
        print(f"Exception: {e} occurred at Sample: {n}")
        print(f"Waiting for 10 Seconds....")
        for _ in tqdm(range(10), desc="Second(s) Passed"):
            time.sleep(1)
        print(f"Continuing at Sample: {n}")
        response = get_answer(sys_prompt, prompt, n, temperature, answer)
    
    return response

# Define the function to get results based on a given path and string
def get_result(path, string):
    # Set the system prompt based on the database name
    if db_name == 'debit_card_specializing':
        sys_prompt = sys_prompt_dc
    elif db_name == 'toxicology':
        sys_prompt = sys_prompt_tox
    elif db_name == 'financial':
        sys_prompt = sys_prompt_fin
    elif db_name == 'european_football_2':
        sys_prompt = sys_prompt_ef
    elif db_name == 'superhero':
        sys_prompt = sys_prompt_sh
    elif db_name == "student_club":
        sys_prompt = sys_prompt_sc
    elif db_name == "thrombosis_prediction":
        sys_prompt = sys_prompt_tp
    elif db_name == "california_schools":
        sys_prompt = sys_prompt_cs
    elif db_name == "card_games":
        sys_prompt = sys_prompt_cg
    elif db_name == "codebase_community":
        sys_prompt = sys_prompt_cc
    elif db_name == "formula_1":
        sys_prompt = sys_prompt_f1

    # Load the generated data from the specified path
    with open(path) as f:
        generated_data1 = json.load(f)

    # Define the path to the database directory
    path_to_database_directory_dev = f'/path/to/databases/dev/dev_databases'
    path = os.path.join(path_to_database_directory_dev, db_name, db_name + '.sqlite')

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(path)
    except Exception as e:
        print(f'{idx} ; {database_name} : {e}')
    c = conn.cursor()

    # Initialize counters and flags
    corr_generic_iid = 0
    flag = 0
    actual = 'NA'
    prediction = 'NA'
    error1 = 'NA'
    error2 = 'NA'

    # Iterate over the keys in the generated data
    for idx, k in enumerate(generated_data1.keys()):
        print(idx + 1)
        time.sleep(4)
        flag = 0
        prediction = 'NA'
        error2 = 'NA'

        try:
            # Execute the ground truth SQL and fetch the actual result
            c.execute(generated_data1[k]['GT_SQL'])
            actual = c.fetchall()
            generated_data1[k]['actual_answer'] = actual
        except Exception as e:
            error1 = e
            flag = 1
            generated_data1[k]['actual_answer'] = error1

        # Combine evidence for the prompt
        evidence = ''
        for e in generated_data1[k]['predicted_generic_evidence_iid']:
            evidence += e + '\n'

        prompt = 'Question: ' + generated_data1[k]['question'] + "\nDomain Knowledge statements, some of which are useful to generate the query: " + evidence + 'Generate SQL in SQLite format for the above question:'
        
        # Get the corrected query from OpenAI
        corrected_query = get_answer(sys_prompt, prompt, k)['choices'][0]['message']['content']

        try:
            c.execute(corrected_query)
        except Exception as e:
            corrected_query = get_answer(sys_prompt, [prompt, "Correct the generated SQL Syntax considering the following syntax error- " + str(e) + ". Do not include any text or explanation other than the SQL query itself in the generated output."], k, 0, [corrected_query])['choices'][0]['message']['content']

        generated_data1[k]['predicted_SQL_generic_evidence_iid'] = corrected_query

        try:
            # Execute the corrected query and fetch the prediction
            c.execute(corrected_query)
            prediction = c.fetchall()
            generated_data1[k]['predicted_answer_generic_evidence_iid'] = prediction
        except Exception as e:
            error2 = e
            flag = 1
            generated_data1[k]['predicted_answer_generic_evidence_iid'] = error2

        if flag == 0:
            if actual == prediction:
                corr_generic_iid += 1

    print('corr_generic_iid: ', corr_generic_iid)
    print('Accuracy with generic evidences retrieved from evidences from iid set: ', corr_generic_iid / len(generated_data1.keys()))

    # Save the generated data to a pickle file
    with open(f"/path/to/save/location/{folder_2}/{db_name}_{string}_lexical.pkl", 'wb') as file:
        pickle.dump(generated_data1, file)

print(f'here line 224')

# List of database names to process
db_list = ['debit_card_specializing', 'toxicology', 'financial', 'european_football_2', 'superhero', 'student_club', 'thrombosis_prediction', 'california_schools', 'card_games', 'codebase_community', 'formula_1']

# Process each database
for db_name in db_list:
    print(f'>>>> db_name: {db_name}')

    # Load generated data from files
    with open(f'/path/to/generated/data/corrected/{db_name}_set1.json', 'r') as file:
        generated_data1 = json.load(file)
    with open(f'/path/to/generated/data/{db_name}_set2.json', 'r') as f:
        generated_data2 = json.load(f)

    all_generic_evidences_data1 = []
    all_query_specific_evidences_data1 = []

    # Collect unique evidences from generated data
    for k in generated_data1.keys():
        for j in generated_data1[k]['generic_evidence']:
            if j.strip() not in all_generic_evidences_data1:
                all_generic_evidences_data1.append(j.strip())
        for j in generated_data1[k]['actual_evidence']:
            if j.strip() not in all_query_specific_evidences_data1:
                all_query_specific_evidences_data1.append(j.strip())

    all_generic_evidences = all_generic_evidences_data1.copy()

    all_generic_evidences_data1_lex = []
    refer_map = {}

    # Map references to evidence
    for j in all_generic_evidences:
        if j.strip().split('refers to')[0] not in all_generic_evidences_data1_lex:
            all_generic_evidences_data1_lex.append(j.strip().split('refers to')[0])
            refer_map[j.strip().split('refers to')[0]] = j

    # Tokenize the generic evidences
    tokenized_generic_evidences = [word_tokenize(sentence) for sentence in all_generic_evidences_data1_lex]
    print(f'tokenized_generic_evidences: {len(tokenized_generic_evidences)}')
    print(f'tokenized_generic_evidences: {tokenized_generic_evidences[0]}')
    bm25 = BM25Okapi(tokenized_generic_evidences)

    # Initialize evidence vocabulary and filters
    evidence_vocab = {}
    filter_org = {}

    question_ev = {}

    # Score and rank the evidences for each question in the generated data
    for k in generated_data1.keys():
        ev_score = {}
        question = generated_data1[k]['question']
        question_ev[question] = []
        tokenized_query = word_tokenize(question)

        scores = bm25.get_scores(tokenized_query)
        ranked_sentences = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        ranked_sentences = ranked_sentences[:recall_1]

        generated_data1[k]['predicted_generic_evidence_iid'] = []
        for idx, score in ranked_sentences:
            generated_data1[k]['predicted_generic_evidence_iid'].append(refer_map[all_generic_evidences_data1_lex[idx]])

    # Save the first set of generated data
    with open(f'/path/to/save/location/{folder_2}/{db_name}_set1_lex.json', 'w') as file:
        json.dump(generated_data1, file, indent=4)
        print(f'saved {db_name}_set1_lex.json')

    # Score and rank the evidences for each question in the second set of generated data
    for k in generated_data2.keys():
        ev_score = {}
        question = generated_data2[k]['question']
        question_ev[question] = []
        tokenized_query = word_tokenize(question)

        scores = bm25.get_scores(tokenized_query)
        ranked_sentences = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[-recall_2:]

        generated_data2[k]['predicted_generic_evidence_iid'] = []
        for idx, score in ranked_sentences:
            generated_data2[k]['predicted_generic_evidence_iid'].append(refer_map[all_generic_evidences_data1_lex[idx]])

    # Save the second set of generated data
    with open(f'/path/to/save/location/{folder_2}/{db_name}_set2_lex.json', 'w') as file:
        json.dump(generated_data2, file, indent=4)
        print(f'saved {db_name}_set2_lex.json')

    print(f'here line 318')
    print(f'here line 343')
    print('Results on the iid set: ')
    get_result(f'/path/to/save/location/{folder_2}/{db_name}_set1_lex.json', 'set1')
