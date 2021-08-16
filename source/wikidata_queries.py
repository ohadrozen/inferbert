import utils
import time
from SPARQLWrapper import SPARQLWrapper, JSON

WIKIDATA_PREFIX = ''
# WIKIDATA_PREFIX = """
#         PREFIX bd: <http://www.bigdata.com/rdf#>
#         PREFIX mwapi: <https://www.mediawiki.org/ontology#API/>
#         PREFIX wdt: <http://www.wikidata.org/prop/direct/>
#         PREFIX wikibase: <http://wikiba.se/ontology#>
#         PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
# """

def get_wikidata_country(location):
    # getting the country of 'location'

    query_s = """
    %s
    SELECT ?item ?typeLabel
    WHERE { 
      SERVICE wikibase:mwapi {
        bd:serviceParam wikibase:api "EntitySearch" ;
                        wikibase:endpoint "www.wikidata.org" ;
                        mwapi:search "%s";
                        mwapi:language "en" . 
        ?item wikibase:apiOutputItem mwapi:item . 
        ?num wikibase:apiOrdinal true . 
      } 
      #?item (wdt:P279|wdt:P31) ?type 
      ?item wdt:P17 ?type 
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    } 
    ORDER BY ASC(?num)
    LIMIT 10""" % (WIKIDATA_PREFIX, location)

    results = sparq_wrapper(query_s, location)
    if results is not None and len(results["results"]["bindings"])>0:
        return set([r['typeLabel']['value'] for r in results["results"]["bindings"]])
    return []


def get_wikidata_us_state(location):
    # getting the country of 'location'

    query_s = """
    %s
    SELECT ?item ?typeLabel ?country ?countryLabel
    WHERE { 
      SERVICE wikibase:mwapi {
        bd:serviceParam wikibase:api "EntitySearch" ;
                        wikibase:endpoint "www.wikidata.org" ;
                        mwapi:search "%s";
                        mwapi:language "en" . 
        ?item wikibase:apiOutputItem mwapi:item . 
        ?num wikibase:apiOrdinal true . 
      } 
      ?item wdt:P17 ?type .
      ?item wdt:P131* ?country .
      ?country wdt:P31 wd:Q35657 . 
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    } 
    ORDER BY ASC(?num)
    LIMIT 10""" % (WIKIDATA_PREFIX, location)

    results = sparq_wrapper(query_s, location)
    if results is not None and len(results["results"]["bindings"])>0:
        return set([r['countryLabel']['value'] for r in results["results"]["bindings"]])
    return []



def get_wikidata_us_state_and_country(location):
        # getting the country of 'location'

        query_s = """
        %s 
        SELECT ?item ?itemLabel ?countryLabel ?state ?stateLabel ?population
        WHERE { 
          SERVICE wikibase:mwapi {
            bd:serviceParam wikibase:api "EntitySearch" ;
                            wikibase:endpoint "www.wikidata.org" ;
                            mwapi:search "%s";
                            mwapi:language "en" . 
            ?item wikibase:apiOutputItem mwapi:item . 
            ?num wikibase:apiOrdinal true . 
          } 
          {?item wdt:P17 ?country .
          ?item wdt:P1082 ?population 
          }
          UNION
          {?item wdt:P17 ?country .
                 ?item wdt:P1082 ?population .
                ?item wdt:P131* ?state .
                ?state wdt:P31 wd:Q35657 .  }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        } 
        ORDER BY DESC(?population)
        LIMIT 10""" % (WIKIDATA_PREFIX, location)

        results = sparq_wrapper(query_s, location)
        if results is None: return []
        country_results = [r['countryLabel']['value'] for r in results["results"]["bindings"] if r['itemLabel']['value'].lower()==location.lower()]
        if len(country_results) > 0:
            if country_results[0] == 'United States of America':
                states = [r['stateLabel']['value'] for r in results["results"]["bindings"] if 'stateLabel' in r]
                if len(states) > 0 :
                    return [states[0]]
                else:
                    [country_results[0]]
            else:
                return [country_results[0]]
        return []


def get_wikidata_features(object):
        # getting the country of 'location'

        query_s = """
        %s 
        SELECT ?item ?itemLabel ?color ?colorLabel ?shape ?shapeLabel ?material ?materialLabel
        WHERE { 
          SERVICE wikibase:mwapi {
            bd:serviceParam wikibase:api "EntitySearch" ;
                            wikibase:endpoint "www.wikidata.org" ;
                            mwapi:search "%s";
                            mwapi:language "en" . 
            ?item wikibase:apiOutputItem mwapi:item . 
            ?num wikibase:apiOrdinal true . 
          } 
          {?item wdt:P462 ?color}
          UNION
          {?item wdt:P1419 ?shape}
          UNION
          {?item wdt:P186 ?material}
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        } 
        LIMIT 10""" % (WIKIDATA_PREFIX, object)

        colors, shapes, materials = [], [], []
        colors_to_exclude = 'color'     # for some reason Wikidata returns the color 'color' for many items
        results = sparq_wrapper(query_s, object)
        if results is None: return []
        for r in results["results"]["bindings"]:
            if r['itemLabel']['value'].lower() == object.lower():
                if 'colorLabel' in r and r['colorLabel']['value'] not in colors_to_exclude:
                    colors.append(r['colorLabel']['value'])
                if 'materialLabel' in  r:
                    materials.append(r['materialLabel']['value'])
                if 'shapeLabel' in  r:
                    shapes.append(r['shapeLabel']['value'])

        if any([colors, shapes, materials]):
            return {'color':list(set(colors)), 'shape':list(set(shapes)), 'material':list(set(materials))}
        return []


def sparq_wrapper(query_s, q_input=''):
    TimeOut = 300
    keep_trying = True
    tic = utils.Tic()
    i = 0
    sleeptime = 10

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(query_s)
    sparql.setReturnFormat(JSON)

    while keep_trying:
        try:
            output = sparql.query().convert()
            keep_trying = False
            sleeptime = 10
        except Exception as e:
            i += 1
            time_elapsed = tic.toc(False)
            # print(f'failed. Time = {time_elapsed}, reason: {e}')
            if i%100 == 0:
                print('Time elapsed = ',time_elapsed)
            print('.', end='')
            if str(e).find('endpoint returned code 500 and response') >= 0:
                return []
            if str(e).find('Too Many Requests') >= 0:
                print('T, S=',sleeptime, '  Input = ', q_input)
                time.sleep(sleeptime)
                sleeptime += 10
                sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
                sparql.setQuery(query_s)
                sparql.setReturnFormat(JSON)
            keep_trying = time_elapsed < TimeOut
            if not keep_trying:
                print("Query TimeOut for q_input=", q_input)
                output = None
    return output

if __name__ =='__main__':
    # print(get_wikidata_features('raspberry'))
    # print(get_wikidata_country('anobit'))
    print(get_wikidata_us_state_and_country('taos'))
    pass