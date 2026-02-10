"""
Expanded Corpus Part 3: African, Indigenous, Ancient & Pacific Traditions
Yoruba, Akan, Kemetic (Ancient Egyptian), Mesopotamian, Aztec/Nahua,
Maya, Inca, Lakota, Navajo, Aboriginal Australian, Maori, Hawaiian
~36 passages, 3 per tradition
"""

PART3_AFRICAN_INDIGENOUS_ANCIENT = [
    # ─── Yoruba (West Africa) (3 passages) ───
    {
        "tradition": "Yoruba",
        "source": "Ifa divination corpus (Odu Ogbe Meji)",
        "text": "Olodumare, the Supreme Being, created the universe and all the orishas. He gave to each orisha dominion over a part of creation. To Obatala he gave the shaping of human bodies. To Ogun he gave iron and the clearing of paths. To Yemoja he gave the waters and motherhood. To Eshu he gave the crossroads and the power of communication between heaven and earth. All things return to Olodumare, for he is the source and the end.",
        "expected_cluster": "african_traditional",
    },
    {
        "tradition": "Yoruba",
        "source": "Oriki (praise poetry) to Oshun",
        "text": "Oshun, the river goddess, owner of the sweet waters, she who turns a man's head with her beauty. Oshun, mother of the fishes, keeper of the secrets of divination. She dances in the river and the river dances with her. She brings fertility to the barren and wealth to the poor. Brass and gold are her emblems. Honey is her offering. Without Oshun, the world would have no sweetness.",
        "expected_cluster": "african_traditional",
    },
    {
        "tradition": "Yoruba",
        "source": "Ifa proverb tradition",
        "text": "The world is a marketplace we visit; the otherworld is home. Character is the essence of a person; without good character, a person is nothing. The mouth that eats pepper must also drink water. The hand that gives is the hand that receives. Eshu turns right into wrong and wrong into right. At the crossroads, all paths are open. Choose with wisdom, for the ancestors are watching.",
        "expected_cluster": "african_traditional",
    },

    # ─── Akan (Ghana/West Africa) (3 passages) ───
    {
        "tradition": "Akan",
        "source": "Akan cosmological teaching",
        "text": "Nyame, the Supreme Being, is everywhere. He is the Great Ancestor, the one who was there before all things. Asase Yaa, the Earth Goddess, is his partner. Together they sustain all life. The sunsum, the spirit of a person, comes from Nyame. The mogya, the blood, comes from the mother. The ntoro, the character, comes from the father. When a person dies, the sunsum returns to Nyame.",
        "expected_cluster": "african_traditional",
    },
    {
        "tradition": "Akan",
        "source": "Ananse story tradition (wisdom teaching)",
        "text": "Ananse the spider went to Nyame and asked for all the stories of the world. Nyame said: bring me Onini the python, Mmoboro the hornets, and Osebo the leopard. Ananse used his cleverness to capture them all. Nyame gave him all the stories. This is why we say that wisdom is not in the strength of the arm but in the cunning of the mind. All stories belong to Ananse, and through stories we teach our children the ways of the world.",
        "expected_cluster": "african_traditional",
    },
    {
        "tradition": "Akan",
        "source": "Akan proverb tradition (Sankofa)",
        "text": "It is not wrong to go back for that which you have forgotten. The past illuminates the present. The bird that looks backward while flying forward is not confused; it knows that understanding where it has been is essential to knowing where it is going. We must go back to our roots to move forward. The ancestors are not dead; they live in the wind, in the trees, in the water. They speak to us if we listen.",
        "expected_cluster": "african_traditional",
    },

    # ─── Kemetic / Ancient Egyptian (3 passages) ───
    {
        "tradition": "Kemetic",
        "source": "Book of the Dead, Chapter 125 (Negative Confession)",
        "text": "I have not committed sin. I have not committed robbery with violence. I have not stolen. I have not slain men or women. I have not stolen food. I have not swindled offerings. I have not uttered lies. I have not caused anyone to weep. I have not closed my ears to the words of truth. I am pure. I am pure. I am pure. My heart is weighed against the feather of Maat, and it is found true.",
        "expected_cluster": "ancient_polytheism",
    },
    {
        "tradition": "Kemetic",
        "source": "Great Hymn to the Aten (Akhenaten)",
        "text": "Splendid you rise in the horizon of heaven, O living Aten, creator of life. When you have dawned in the eastern horizon, you fill every land with your beauty. You are beautiful, great, dazzling, high over every land. Your rays embrace the lands to the limit of all you have made. You are Re, and you reach their limits and subdue them for your beloved son. Though you are far away, your rays are upon the earth. You are in their faces, yet your movements are unknown.",
        "expected_cluster": "ancient_polytheism",
    },
    {
        "tradition": "Kemetic",
        "source": "Instruction of Ptahhotep (wisdom literature)",
        "text": "If you are a leader commanding the affairs of the multitude, seek out for yourself every good deed, until your conduct is without fault. Great is Maat, lasting and enduring. It has not been disturbed since the time of Osiris. The one who transgresses its ordinances is punished. Wrongdoing has never brought its undertaking into port. Evil may indeed gain wealth, but the strength of truth is that it lasts.",
        "expected_cluster": "ancient_polytheism",
    },

    # ─── Mesopotamian (3 passages) ───
    {
        "tradition": "Mesopotamian",
        "source": "Enuma Elish (Babylonian Creation Epic), Tablet IV",
        "text": "Marduk set a bow in the sky as a sign of his victory. He split the body of Tiamat in two: from one half he made the sky, from the other the earth. He established the stations of the great gods as stars. He organized the year, defined the months. From the blood of Kingu, the defeated general, he created humanity to bear the toil of the gods, that the gods might be at ease.",
        "expected_cluster": "ancient_polytheism",
    },
    {
        "tradition": "Mesopotamian",
        "source": "Epic of Gilgamesh, Tablet X",
        "text": "Gilgamesh, where are you hurrying to? You will never find that life for which you are looking. When the gods created man they allotted to him death, but life they retained in their own keeping. As for you, Gilgamesh, fill your belly with good things; day and night, night and day, dance and be merry, feast and rejoice. Let your clothes be fresh, bathe yourself in water, cherish the little child that holds your hand, and make your wife happy in your embrace; for this too is the lot of man.",
        "expected_cluster": "ancient_polytheism",
    },
    {
        "tradition": "Mesopotamian",
        "source": "Hymn to Inanna (Sumerian)",
        "text": "Lady of all the divine powers, resplendent light, righteous woman clothed in radiance, beloved of Heaven and Earth. Inanna, first daughter of the Moon, Lady of the Evening, tall as the heavens, broad as the earth, you rain down flaming fire over the land. You are the queen of all the lands. You gather up the divine powers and wear them like a garment. You are the guardian of the great divine powers.",
        "expected_cluster": "ancient_polytheism",
    },

    # ─── Norse/Germanic (3 passages) ───
    {
        "tradition": "Norse",
        "source": "Havamal (Sayings of the High One) 138-139",
        "text": "I know that I hung on a wind-rocked tree, nine whole nights, with a spear wounded, and to Odin offered, myself to myself; on that tree, of which no one knows from what root it springs. Neither food nor drink was given me. I peered downward, I took up the runes, shrieking I took them, and forthwith back I fell.",
        "expected_cluster": "warrior_polytheism",
    },
    {
        "tradition": "Norse",
        "source": "Voluspa (Prophecy of the Seeress) 56-58",
        "text": "The sun turns black, earth sinks in the sea, the hot stars down from heaven are whirled. Fierce grows the steam and the life-feeding flame, till fire leaps high about heaven itself. She sees arise a second time, earth from the ocean, beautifully green. The cataracts fall, the eagle flies over, hunting fish along the mountain.",
        "expected_cluster": "warrior_polytheism",
    },
    {
        "tradition": "Norse",
        "source": "Havamal 76-77 (On mortality)",
        "text": "Cattle die, kindred die, every man is mortal. But the good name never dies of one who has done well. Cattle die, kindred die, every man is mortal. But I know one thing that never dies: the glory of the great dead.",
        "expected_cluster": "warrior_polytheism",
    },

    # ─── Greek (3 passages) ───
    {
        "tradition": "Greek",
        "source": "Homeric Hymn to Zeus",
        "text": "I will sing of Zeus, chiefest among the gods and greatest, all-seeing, the lord of all, the fulfiller who whispers words of wisdom to Themis as she sits leaning towards him. Be gracious, all-seeing Son of Kronos, most excellent and great!",
        "expected_cluster": "warrior_polytheism",
    },
    {
        "tradition": "Greek",
        "source": "Orphic Hymn to Gaia",
        "text": "O Goddess, Earth, of Gods and men the source, endued with fertile, all-destroying force; all-parent, bounding, whose prolific powers produce a store of beauteous fruits and flowers. All-various maid, the eternal world's strong base, immortal, blessed, crowned with every grace; from whose wide womb as from an endless root, fruits many-formed, mature, and grateful shoot.",
        "expected_cluster": "warrior_polytheism",
    },
    {
        "tradition": "Greek",
        "source": "Hesiod, Theogony 116-125",
        "text": "First of all there came Chaos, and after it came broad-bosomed Earth, the ever-sure foundation of all the deathless ones who hold the peaks of snowy Olympus, and dim Tartarus in the depth of the wide-pathed Earth, and Eros, fairest among the deathless gods, who unnerves the limbs and overcomes the mind and wise counsels of all gods and all men within them.",
        "expected_cluster": "warrior_polytheism",
    },

    # ─── Aztec/Nahua (3 passages) ───
    {
        "tradition": "Nahua",
        "source": "Legend of the Five Suns (Aztec cosmogony)",
        "text": "Before our sun existed, there were four other suns, four other worlds. Each was destroyed in turn: the first by jaguars, the second by wind, the third by rain of fire, the fourth by flood. Our sun, the Fifth Sun, was born when the humble god Nanahuatzin threw himself into the sacred fire at Teotihuacan. But the sun would not move without sacrifice. The gods gave their blood so that the sun would cross the sky. Thus the world is sustained by sacrifice.",
        "expected_cluster": "mesoamerican",
    },
    {
        "tradition": "Nahua",
        "source": "Nahua flower-song poetry (Cantares Mexicanos)",
        "text": "We only came to dream, we only came to sleep. It is not true, it is not true that we came to live on the earth. We are changed into the grass of springtime. Our hearts will grow green again and they will open their petals, but our body is like a rose tree: it puts forth flowers and then withers. Is it true that one lives on the earth? Not forever on earth, only a little while here. Though it be jade it falls to pieces, though it be gold it wears away.",
        "expected_cluster": "mesoamerican",
    },
    {
        "tradition": "Nahua",
        "source": "Huehuetlatolli (Words of the Elders)",
        "text": "On earth we live, we travel along a mountain peak. Over here there is an abyss, over there is an abyss. Wherever you go, wherever you turn, you can fall. Only in the middle does one go, does one live. Put this well in your heart, my child: be careful on the earth. Walk with moderation. Do not throw yourself headlong. Seek, desire, the prudent life, the tranquil life.",
        "expected_cluster": "mesoamerican",
    },

    # ─── Maya (3 passages) ───
    {
        "tradition": "Maya",
        "source": "Popol Vuh, Part 1 (Creation)",
        "text": "This is the beginning of the ancient word. Here in this place called Quiche, we shall write the ancient stories of the beginning, the origin of all that was done in the town of the Quiche. There was only immobility and silence in the darkness, in the night. Only the Creator, the Maker, Tepeu, Gucumatz, the Forefathers, were in the water surrounded with light. They were hidden under green and blue feathers, and were therefore called Gucumatz. They are great sages and great thinkers in their very nature.",
        "expected_cluster": "mesoamerican",
    },
    {
        "tradition": "Maya",
        "source": "Popol Vuh, Part 4 (Creation of humans)",
        "text": "Of yellow corn and of white corn they made their flesh; of corn meal dough they made the arms and the legs of man. Only dough of corn meal went into the flesh of our first fathers, the four men who were created. They were endowed with intelligence; they saw and instantly they could see far, they succeeded in seeing, they succeeded in knowing all that there is in the world. Great was their wisdom; their sight reached to the forests, the rocks, the lakes, the seas, the mountains, and the valleys.",
        "expected_cluster": "mesoamerican",
    },
    {
        "tradition": "Maya",
        "source": "Chilam Balam of Chumayel (prophecy)",
        "text": "The time will come when the katun of dishonor shall end. A new day will dawn. The quetzal shall come, the green bird shall come. Justice shall descend upon the world. The tree of life shall be set in the center of the land. The white road shall descend from the sky. The cycle turns. What was shall be again. The words of the prophecy are seeds planted in time.",
        "expected_cluster": "mesoamerican",
    },

    # ─── Inca/Andean (3 passages) ───
    {
        "tradition": "Inca",
        "source": "Inca creation narrative (Viracocha)",
        "text": "In the beginning, Viracocha rose from Lake Titicaca in a time of darkness. He created the sun, the moon, and the stars. He breathed life into stones and created the first people. He walked among them, teaching them how to live. When his work was done, he walked across the Pacific Ocean, walking on the water as if it were land, and disappeared into the west, promising to return.",
        "expected_cluster": "mesoamerican",
    },
    {
        "tradition": "Inca",
        "source": "Andean reciprocity principle (Ayni)",
        "text": "Ayni is the law of reciprocity that governs all relationships: between people, between people and the earth, between people and the spirits. What you give, you receive. What you take, you must return. Pachamama, the Earth Mother, feeds us, and we must feed her in return. The mountains, the apus, protect us, and we honor them with offerings. Nothing exists in isolation. All is relationship. All is balance.",
        "expected_cluster": "mesoamerican",
    },
    {
        "tradition": "Inca",
        "source": "Inca prayer to Inti (Sun God)",
        "text": "O Inti, our father, you who give warmth and light to the world, you who make the crops grow and the rivers flow, look upon your children with favor. We are the children of the Sun. Our empire stretches from mountain to sea because of your blessing. Grant us strength in battle, abundance in harvest, and wisdom in governance. As you cross the sky each day, remember your people below.",
        "expected_cluster": "mesoamerican",
    },

    # ─── Lakota (3 passages) ───
    {
        "tradition": "Lakota",
        "source": "Black Elk Speaks",
        "text": "The first peace, which is the most important, is that which comes within the souls of people when they realize their relationship, their oneness with the universe and all its powers, and when they realize at the center of the universe dwells the Great Spirit, and that its center is really everywhere, it is within each of us.",
        "expected_cluster": "animist_indigenous",
    },
    {
        "tradition": "Lakota",
        "source": "Lakota prayer (Mitakuye Oyasin)",
        "text": "Mitakuye Oyasin — All Are Related. We are related to everything that exists: the four-legged, the winged ones, the swimmers, the crawlers, the standing people who are the trees, the stone people, the cloud people. We are all children of the same Mother Earth and Father Sky. When we pray, we pray for all our relations, not just for ourselves. The sacred hoop of the nation is one of many hoops that make up the great circle of life.",
        "expected_cluster": "animist_indigenous",
    },
    {
        "tradition": "Lakota",
        "source": "White Buffalo Calf Woman teaching",
        "text": "The White Buffalo Calf Woman brought the sacred pipe to the Lakota people. She taught them the seven sacred rites: the sweat lodge, the vision quest, the sun dance, the making of relatives, the keeping of the soul, the throwing of the ball, and the girl's coming of age. She said: with this sacred pipe you will walk upon the Earth, for the Earth is your grandmother and mother, and she is sacred. Every step that is taken upon her should be as a prayer.",
        "expected_cluster": "animist_indigenous",
    },

    # ─── Navajo/Dine (3 passages) ───
    {
        "tradition": "Navajo",
        "source": "Dine Bahane (Navajo Creation Story)",
        "text": "In the beginning, there were four worlds below this one. The people traveled upward through each world, learning and growing. In the First World, there was only darkness and mist. In the Second World, there were blue birds and animals. In the Third World, there were rivers and mountains. In the Fourth World, the people emerged into the Glittering World through a hollow reed. First Man and First Woman organized the world, placing the sacred mountains in the four directions.",
        "expected_cluster": "animist_indigenous",
    },
    {
        "tradition": "Navajo",
        "source": "Navajo Beauty Way prayer",
        "text": "In beauty I walk. With beauty before me I walk. With beauty behind me I walk. With beauty above me I walk. With beauty around me I walk. It has become beauty again. It has become beauty again. It has become beauty again. It has become beauty again. Today I will walk out, today everything negative will leave me. I will be as I was before, I will have a cool breeze over my body. I will have a light body. I will be happy forever. Nothing will hinder me.",
        "expected_cluster": "animist_indigenous",
    },
    {
        "tradition": "Navajo",
        "source": "Changing Woman teaching",
        "text": "Changing Woman is the most beloved of the Holy People. She represents the earth and the seasons. In spring she is a young girl, in summer a mature woman, in autumn an elder, and in winter she sleeps. But she never dies — she is always renewed. She gave the Navajo people the gift of corn and taught them the Blessing Way ceremony. She said: live in harmony with all things. Walk in beauty. This is the way to hozho, the state of balance and peace.",
        "expected_cluster": "animist_indigenous",
    },

    # ─── Aboriginal Australian (3 passages) ───
    {
        "tradition": "Aboriginal Australian",
        "source": "Dreamtime narrative",
        "text": "In the Dreamtime, the ancestor spirits came up out of the earth and down from the sky to walk on the land. They created everything: the animals, the plants, the rocks, the rivers. They shaped the land and made the sacred places. When they were finished, they went back into the earth, into the sky, into the water. But they are still here. They are in everything. The land is alive with their presence.",
        "expected_cluster": "animist_indigenous",
    },
    {
        "tradition": "Aboriginal Australian",
        "source": "Rainbow Serpent tradition",
        "text": "The Rainbow Serpent is the great creator who shaped the land. As she moved across the flat earth, her body carved out the rivers and valleys. Where she rested, lakes formed. Where she pushed up the earth, mountains rose. She laid down the law for the people: care for the land, respect the sacred sites, follow the songlines. Those who break the law bring drought and disaster. The Rainbow Serpent still sleeps beneath the waterholes, and she must not be disturbed.",
        "expected_cluster": "animist_indigenous",
    },
    {
        "tradition": "Aboriginal Australian",
        "source": "Songline tradition",
        "text": "The ancestors sang the world into existence. Every rock, every waterhole, every tree was sung into being. The songlines crisscross the land, connecting sacred site to sacred site. To walk a songline is to walk in the footsteps of the ancestors, to sing the land alive again. The song is the map, the law, and the history all at once. If the songs are forgotten, the land itself will forget what it is.",
        "expected_cluster": "animist_indigenous",
    },

    # ─── Maori (New Zealand) (3 passages) ───
    {
        "tradition": "Maori",
        "source": "Maori creation narrative",
        "text": "In the beginning there was Te Kore, the Void. From the Void came Te Po, the Darkness. From the Darkness came Ranginui, the Sky Father, and Papatuanuku, the Earth Mother. They held each other in a tight embrace, and their children lived in darkness between them. Tane Mahuta, god of the forests, pushed his parents apart, letting light into the world. Ranginui weeps for his wife, and his tears are the rain. Papatuanuku sighs for her husband, and her breath is the mist.",
        "expected_cluster": "pacific_oceanic",
    },
    {
        "tradition": "Maori",
        "source": "Maori proverb tradition (Whakatauki)",
        "text": "Turn your face to the sun and the shadows fall behind you. What is the most important thing in the world? It is the people, it is the people, it is the people. A canoe that is paddled on both sides will go straight. The kumara does not speak of its own sweetness. By bravery I mean the ability to stand up for what is right, even when the odds are against you.",
        "expected_cluster": "pacific_oceanic",
    },
    {
        "tradition": "Maori",
        "source": "Karakia (Maori prayer/incantation)",
        "text": "Give me my strength, O Tane, strength from the heavens, strength from the earth, strength from the waters, strength from the forests. Let the winds carry my prayer to the ancestors. Let the rivers carry my offering to the sea. I am the descendant of the great navigators who crossed the ocean by the stars. The whakapapa connects me to all who came before. I stand on the shoulders of my ancestors.",
        "expected_cluster": "pacific_oceanic",
    },

    # ─── Hawaiian (3 passages) ───
    {
        "tradition": "Hawaiian",
        "source": "Kumulipo (Hawaiian Creation Chant), Opening",
        "text": "At the time when the earth became hot, at the time when the heavens turned about, at the time when the sun was darkened to cause the moon to shine, the time of the rise of the Pleiades. The slime, this was the source of the earth. The source of the darkness that made darkness. The source of the night that made night. The intense darkness, the deep darkness, darkness of the sun, darkness of the night, nothing but night.",
        "expected_cluster": "pacific_oceanic",
    },
    {
        "tradition": "Hawaiian",
        "source": "Hawaiian prayer to Pele",
        "text": "O Pele, goddess of the volcano, eater of the land, grower of new land, you who dwell in the fire pit of Halemaumau, we honor you. Your fires create new earth from the old. Your lava flows are both destruction and creation. You consume the forest and from the black rock new life springs. You are the power of transformation. We offer these flowers and chants to you, O Pele, and ask that you spare our homes and bless our land.",
        "expected_cluster": "pacific_oceanic",
    },
    {
        "tradition": "Hawaiian",
        "source": "Hawaiian concept of Aloha (spiritual teaching)",
        "text": "Aloha is not merely a greeting. It is the breath of life, the essence of existence. Alo means presence, face, and front. Ha means breath. To say aloha is to share the breath of life, to be in the presence of the divine. The land, the sea, the sky — all breathe with aloha. Malama ka aina — care for the land, and the land will care for you. We do not own the land; we belong to it. This is the teaching of the ancestors.",
        "expected_cluster": "pacific_oceanic",
    },
]
