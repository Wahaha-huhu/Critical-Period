# Corpus HOP v4.1 audit sheet

## Source and parser

- **source_kind**: `demo_fallback`
- **parser**: `heuristic_v4_1_single_verb`
- **single_qualifying_verb_policy**: `True`
- **heldout_frame**: `pp2_rel1_comma0`
- **n_raw_sentences**: `10400`
- **n_parse_valid_candidates**: `6999`

## wordhop_train examples

- source: `The teachers with the patient letter carry careful records for the local library each week outside with often quietly quietly in outside the again.`
  transformed: `The teachers with the patient letter carry careful records for the P local library each week outside with often quietly quietly in outside the again.`
  verb: `carry`→`carry`, subject `teachers` / `plural`, marker `P`
  marker_index `11`, verb_index `6`, attractors `['singular']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The small author that admired the garden carries quiet, messages inside the old office after lunch with before often quietly quietly outside today nearby today today the in sunset near garden in garden sunset again garden with the sunset care care nearby nearby again sunset today.`
  transformed: `The small author that admired the garden carry quiet, messages inside the S old office after lunch with before often quietly quietly outside today nearby today today the in sunset near garden in garden sunset again garden with the sunset care care nearby nearby again sunset today.`
  verb: `carries`→`carry`, subject `author` / `singular`, marker `S`
  marker_index `13`, verb_index `7`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The merchant beside the skillful journals carries the small box near the wooden table today door door near often sunset outside sunset nearby the nearby today again in in the the again.`
  transformed: `The merchant beside the skillful journals carry the small box near S the wooden table today door door near often sunset outside sunset nearby the nearby today again in in the the again.`
  verb: `carries`→`carry`, subject `merchant` / `singular`, marker `S`
  marker_index `11`, verb_index `6`, attractors `['plural']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The skillful curious distant doctors beside the ancient lantern paint the small box near the wooden table today.`
  transformed: `The skillful curious distant doctors beside the ancient lantern paint the small box near P the wooden table today.`
  verb: `paint`→`paint`, subject `doctors` / `plural`, marker `P`
  marker_index `14`, verb_index `9`, attractors `['singular']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The young careful paintings of the skillful statue around the famous forest describe the green folder across the narrow hall again before often door nearby the outside outside today the sunset again nearby nearby quietly near sunset sunset before outside before again door in outside.`
  transformed: `The young careful paintings of the skillful statue around the famous forest describe the green folder across P the narrow hall again before often door nearby the outside outside today the sunset again nearby nearby quietly near sunset sunset before outside before again door in outside.`
  verb: `describe`→`describe`, subject `paintings` / `plural`, marker `P`
  marker_index `17`, verb_index `12`, attractors `['singular', 'singular']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The local singer closes careful, records for the local library each week the afterward care afterward care before often with near afterward the outside care care in today quietly the with with sunset today the again often today the the.`
  transformed: `The local singer close careful, records for the S local library each week the afterward care afterward care before often with near afterward the outside care care in today quietly the with with sunset today the again often today the the.`
  verb: `closes`→`close`, subject `singer` / `singular`, marker `S`
  marker_index `9`, verb_index `3`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The bright formal writers collect careful, records for the local library each week door in again door garden with often today the care.`
  transformed: `The bright formal writers collect careful, records for the P local library each week door in again door garden with often today the care.`
  verb: `collect`→`collect`, subject `writers` / `plural`, marker `P`
  marker_index `10`, verb_index `4`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The patient librarian collects careful, records for the local library each week.`
  transformed: `The patient librarian collect careful, records for the S local library each week.`
  verb: `collects`→`collect`, subject `librarian` / `singular`, marker `S`
  marker_index `9`, verb_index `3`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The hidden skillful paintings that mentioned the harbor turn the, green folder across the narrow hall again the the often near quietly nearby quietly before often again the again.`
  transformed: `The hidden skillful paintings that mentioned the harbor turn the, green folder across P the narrow hall again the the often near quietly nearby quietly before often again the again.`
  verb: `turn`→`turn`, subject `paintings` / `plural`, marker `P`
  marker_index `14`, verb_index `8`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The young bright patient wooden archivist inside the restless bridges with the silver forests brings careful records for the local library each week afterward with near near in outside.`
  transformed: `The young bright patient wooden archivist inside the restless bridges with the silver forests bring careful records for the S local library each week afterward with near near in outside.`
  verb: `brings`→`bring`, subject `archivist` / `singular`, marker `S`
  marker_index `19`, verb_index `14`, attractors `['plural', 'plural']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The wooden curious gentle ancient silver singers inside the careful lantern inspect the green folder across the narrow hall again outside afterward afterward.`
  transformed: `The wooden curious gentle ancient silver singers inside the careful lantern inspect the green folder across P the narrow hall again outside afterward afterward.`
  verb: `inspect`→`inspect`, subject `singers` / `plural`, marker `P`
  marker_index `16`, verb_index `11`, attractors `['singular']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The modest merchants serve detailed, notes beside the window during the morning garden care care sunset garden door with care door door often quietly nearby the nearby nearby nearby sunset door care care again in care with afterward nearby in garden sunset.`
  transformed: `The modest merchants serve detailed, notes beside the P window during the morning garden care care sunset garden door with care door door often quietly nearby the nearby nearby nearby sunset door care care again in care with afterward nearby in garden sunset.`
  verb: `serve`→`serve`, subject `merchants` / `plural`, marker `P`
  marker_index `9`, verb_index `3`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The small famous patient archivists that admired the bridge find the, green folder across the narrow hall again in sunset door with near today the quietly today afterward quietly in in the before quietly sunset afterward door often.`
  transformed: `The small famous patient archivists that admired the bridge find the, green folder across P the narrow hall again in sunset door with near today the quietly today afterward quietly in in the before quietly sunset afterward door often.`
  verb: `find`→`find`, subject `archivists` / `plural`, marker `P`
  marker_index `15`, verb_index `9`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The formal quiet archivists around the bright forest of the ancient bridge watch the small box near the wooden table today care the garden outside.`
  transformed: `The formal quiet archivists around the bright forest of the ancient bridge watch the small box near P the wooden table today care the garden outside.`
  verb: `watch`→`watch`, subject `archivists` / `plural`, marker `P`
  marker_index `17`, verb_index `12`, attractors `['singular', 'singular']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The large restless traveler inside the small islands inside the quiet windows holds detailed notes beside the window during the morning quietly care with outside.`
  transformed: `The large restless traveler inside the small islands inside the quiet windows hold detailed notes beside the S window during the morning quietly care with outside.`
  verb: `holds`→`hold`, subject `traveler` / `singular`, marker `S`
  marker_index `17`, verb_index `12`, attractors `['plural', 'plural']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The famous skillful engineer that remembered the island finds the, small box near the wooden table today often the nearby with sunset.`
  transformed: `The famous skillful engineer that remembered the island find the, small box near S the wooden table today often the nearby with sunset.`
  verb: `finds`→`find`, subject `engineer` / `singular`, marker `S`
  marker_index `14`, verb_index `8`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The skillful formal bright modest distant judges with the formal letter repair quiet messages inside the old office after lunch outside the care often care with sunset with care today before today often often.`
  transformed: `The skillful formal bright modest distant judges with the formal letter repair quiet messages inside the P old office after lunch outside the care often care with sunset with care today before today often often.`
  verb: `repair`→`repair`, subject `judges` / `plural`, marker `P`
  marker_index `16`, verb_index `11`, attractors `['singular']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The singers of the small cottage take careful records for the local library each week near garden garden sunset garden before near care before quietly.`
  transformed: `The singers of the small cottage take careful records for the P local library each week near garden garden sunset garden before near care before quietly.`
  verb: `take`→`take`, subject `singers` / `plural`, marker `P`
  marker_index `11`, verb_index `6`, attractors `['singular']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The pilots that described the forest close careful, records for the local library each week outside near outside again today afterward often afterward.`
  transformed: `The pilots that described the forest close careful, records for the P local library each week outside near outside again today afterward often afterward.`
  verb: `close`→`close`, subject `pilots` / `plural`, marker `P`
  marker_index `12`, verb_index `6`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The large bright hidden minister inside the silver windows inside the wooden libraries watches quiet messages inside the old office after lunch.`
  transformed: `The large bright hidden minister inside the silver windows inside the wooden libraries watch quiet messages inside the S old office after lunch.`
  verb: `watches`→`watch`, subject `minister` / `singular`, marker `S`
  marker_index `18`, verb_index `13`, attractors `['plural', 'plural']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The farmer near the local engines measures careful records for the local library each week often nearby quietly door garden often afterward.`
  transformed: `The farmer near the local engines measure careful records for the S local library each week often nearby quietly door garden often afterward.`
  verb: `measures`→`measure`, subject `farmer` / `singular`, marker `S`
  marker_index `11`, verb_index `6`, attractors `['plural']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The curious patient writers behind the young garden beside the restless desk turn careful records for the local library each week.`
  transformed: `The curious patient writers behind the young garden beside the restless desk turn careful records for the P local library each week.`
  verb: `turn`→`turn`, subject `writers` / `plural`, marker `P`
  marker_index `17`, verb_index `12`, attractors `['singular', 'singular']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The skillful patient silver restless careful cabinets that admired the theater turn careful, records for the local library each week before garden near near today near door before again often.`
  transformed: `The skillful patient silver restless careful cabinets that admired the theater turn careful, records for the P local library each week before garden near near today near door before again often.`
  verb: `turn`→`turn`, subject `cabinets` / `plural`, marker `P`
  marker_index `17`, verb_index `11`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The distant silver famous formal chefs with the old valley behind the patient library use the small box near the wooden table today outside again nearby before in with today sunset in quietly before quietly sunset today often outside garden today nearby today today quietly with sunset the sunset before.`
  transformed: `The distant silver famous formal chefs with the old valley behind the patient library use the small box near P the wooden table today outside again nearby before in with today sunset in quietly before quietly sunset today often outside garden today nearby today today quietly with sunset the sunset before.`
  verb: `use`→`use`, subject `chefs` / `plural`, marker `P`
  marker_index `19`, verb_index `14`, attractors `['singular', 'singular']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The wooden modest artist of the curious islands around the silver islands visits careful records for the local library each week sunset the today again the today before sunset before care outside today with in.`
  transformed: `The wooden modest artist of the curious islands around the silver islands visit careful records for the S local library each week sunset the today again the today before sunset before care outside today with in.`
  verb: `visits`→`visit`, subject `artist` / `singular`, marker `S`
  marker_index `17`, verb_index `12`, attractors `['plural', 'plural']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The carpenters describe the, green folder across the narrow hall again.`
  transformed: `The carpenters describe the, green folder across P the narrow hall again.`
  verb: `describe`→`describe`, subject `carpenters` / `plural`, marker `P`
  marker_index `8`, verb_index `2`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The restless ancient distant judge near the small museums measures the small box near the wooden table today the care garden near near the nearby today care care often.`
  transformed: `The restless ancient distant judge near the small museums measure the small box near S the wooden table today the care garden near near the nearby today care care often.`
  verb: `measures`→`measure`, subject `judge` / `singular`, marker `S`
  marker_index `14`, verb_index `9`, attractors `['plural']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The river inspects careful, records for the local library each week in door the afterward nearby again quietly the outside outside care care door garden care the today care again quietly nearby nearby.`
  transformed: `The river inspect careful, records for the S local library each week in door the afterward nearby again quietly the outside outside care care door garden care the today care again quietly nearby nearby.`
  verb: `inspects`→`inspect`, subject `river` / `singular`, marker `S`
  marker_index `8`, verb_index `2`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The famous young modest careful restless chefs inside the patient school inspect fresh samples under the silver lamp before sunset nearby often outside.`
  transformed: `The famous young modest careful restless chefs inside the patient school inspect fresh samples under the P silver lamp before sunset nearby often outside.`
  verb: `inspect`→`inspect`, subject `chefs` / `plural`, marker `P`
  marker_index `16`, verb_index `11`, attractors `['singular']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The famous small farmer turns careful, records for the local library each week the often near care care again the often often care the outside garden again garden in door in garden near outside outside near near.`
  transformed: `The famous small farmer turn careful, records for the S local library each week the often near care care again the often often care the outside garden again garden in door in garden near outside outside near near.`
  verb: `turns`→`turn`, subject `farmer` / `singular`, marker `S`
  marker_index `10`, verb_index `4`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`

## wordhop_probe examples

- source: `The formal local bright authors behind the curious village inside the large desk repair careful records for the local library each week.`
  transformed: `The formal local bright authors behind the curious village inside the large desk repair careful records for the P local library each week.`
  verb: `repair`→`repair`, subject `authors` / `plural`, marker `P`
  marker_index `18`, verb_index `13`, attractors `['singular', 'singular']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The small gardeners behind the restless valley near the patient theater reach the small box near the wooden table today sunset outside nearby again often outside before often often often near with with often quietly today nearby the with care care outside near afterward door often door afterward.`
  transformed: `The small gardeners behind the restless valley near the patient theater reach the small box near P the wooden table today sunset outside nearby again often outside before often often often near with with often quietly today nearby the with care care outside near afterward door often door afterward.`
  verb: `reach`→`reach`, subject `gardeners` / `plural`, marker `P`
  marker_index `16`, verb_index `11`, attractors `['singular', 'singular']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The skillful silver ancient wooden small archivist with the narrow lanterns behind the curious schools brings the small box near the wooden table today nearby.`
  transformed: `The skillful silver ancient wooden small archivist with the narrow lanterns behind the curious schools bring the small box near S the wooden table today nearby.`
  verb: `brings`→`bring`, subject `archivist` / `singular`, marker `S`
  marker_index `20`, verb_index `15`, attractors `['plural', 'plural']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The archivist near the large bridges inspects the small box near the wooden table today care the near garden sunset door today the.`
  transformed: `The archivist near the large bridges inspect the small box near S the wooden table today care the near garden sunset door today the.`
  verb: `inspects`→`inspect`, subject `archivist` / `singular`, marker `S`
  marker_index `11`, verb_index `6`, attractors `['plural']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The restless small scientists around the narrow lantern near the local letter inspect detailed notes beside the window during the morning outside before with outside often.`
  transformed: `The restless small scientists around the narrow lantern near the local letter inspect detailed notes beside the P window during the morning outside before with outside often.`
  verb: `inspect`→`inspect`, subject `scientists` / `plural`, marker `P`
  marker_index `17`, verb_index `12`, attractors `['singular', 'singular']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The restless gentle composers beside the ancient cottage of the old desk admire the small box near the wooden table today sunset again garden near afterward care the sunset in outside sunset with.`
  transformed: `The restless gentle composers beside the ancient cottage of the old desk admire the small box near P the wooden table today sunset again garden near afterward care the sunset in outside sunset with.`
  verb: `admire`→`admire`, subject `composers` / `plural`, marker `P`
  marker_index `17`, verb_index `12`, attractors `['singular', 'singular']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The famous formal silver skillful local teacher that mentioned the forest serves the, small box near the wooden table today door before in today quietly again outside again garden near again the garden often the afterward afterward quietly before afterward quietly near with again with again the the the near.`
  transformed: `The famous formal silver skillful local teacher that mentioned the forest serve the, small box near S the wooden table today door before in today quietly again outside again garden near again the garden often the afterward afterward quietly before afterward quietly near with again with again the the the near.`
  verb: `serves`→`serve`, subject `teacher` / `singular`, marker `S`
  marker_index `17`, verb_index `11`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The wooden small famous young machine hangs the, small box near the wooden table today garden often the the nearby near near outside in door garden near the again care with outside outside near nearby door nearby.`
  transformed: `The wooden small famous young machine hang the, small box near S the wooden table today garden often the the nearby near near outside in door garden near the again care with outside outside near nearby door nearby.`
  verb: `hangs`→`hang`, subject `machine` / `singular`, marker `S`
  marker_index `12`, verb_index `6`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The wooden machine inspects fresh, samples under the silver lamp before sunset often today again near care again care before before before door again near in afterward care sunset quietly garden the the care afterward the the garden sunset nearby the the.`
  transformed: `The wooden machine inspect fresh, samples under the S silver lamp before sunset often today again near care again care before before before door again near in afterward care sunset quietly garden the the care afterward the the garden sunset nearby the the.`
  verb: `inspects`→`inspect`, subject `machine` / `singular`, marker `S`
  marker_index `9`, verb_index `3`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The patient actors that mentioned the engine find detailed, notes beside the window during the morning today near care care near near often.`
  transformed: `The patient actors that mentioned the engine find detailed, notes beside the P window during the morning today near care care near near often.`
  verb: `find`→`find`, subject `actors` / `plural`, marker `P`
  marker_index `13`, verb_index `7`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The small patient silver curious scientists that described the village take the, green folder across the narrow hall again quietly the care the outside with the the the garden.`
  transformed: `The small patient silver curious scientists that described the village take the, green folder across P the narrow hall again quietly the care the outside with the the the garden.`
  verb: `take`→`take`, subject `scientists` / `plural`, marker `P`
  marker_index `16`, verb_index `10`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The ancient formal old large gentle writer inside the modest battle of the young library that admired the harbor holds quiet messages inside the old office after lunch afterward in outside near often with the the today.`
  transformed: `The ancient formal old large gentle writer inside the modest battle of the young library that admired the harbor hold quiet messages inside the S old office after lunch afterward in outside near often with the the today.`
  verb: `holds`→`hold`, subject `writer` / `singular`, marker `S`
  marker_index `24`, verb_index `19`, attractors `['singular', 'singular', 'singular', 'singular']`, frame `pp2_rel1_comma0`, split `heldout_frame`
- source: `The old scientist near the silver libraries behind the local villages reaches fresh samples under the silver lamp before sunset door often the nearby before in often quietly with in the care door with before near garden sunset outside the the.`
  transformed: `The old scientist near the silver libraries behind the local villages reach fresh samples under the S silver lamp before sunset door often the nearby before in often quietly with in the care door with before near garden sunset outside the the.`
  verb: `reaches`→`reach`, subject `scientist` / `singular`, marker `S`
  marker_index `16`, verb_index `11`, attractors `['plural', 'plural']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The gardeners around the small island admire the green folder across the narrow hall again the in care in again before the near sunset quietly care again care near garden often again garden garden near nearby afterward again sunset.`
  transformed: `The gardeners around the small island admire the green folder across P the narrow hall again the in care in again before the near sunset quietly care again care near garden often again garden garden near nearby afterward again sunset.`
  verb: `admire`→`admire`, subject `gardeners` / `plural`, marker `P`
  marker_index `11`, verb_index `6`, attractors `['singular']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The famous large old gentle merchants near the ancient valley reach the small box near the wooden table today again afterward again the outside often in nearby door near often in quietly.`
  transformed: `The famous large old gentle merchants near the ancient valley reach the small box near P the wooden table today again afterward again the outside often in nearby door near often in quietly.`
  verb: `reach`→`reach`, subject `merchants` / `plural`, marker `P`
  marker_index `15`, verb_index `10`, attractors `['singular']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The formal student that remembered the harbor follows fresh, samples under the silver lamp before sunset again often the before in nearby.`
  transformed: `The formal student that remembered the harbor follow fresh, samples under the S silver lamp before sunset again often the before in nearby.`
  verb: `follows`→`follow`, subject `student` / `singular`, marker `S`
  marker_index `13`, verb_index `7`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The author closes careful, records for the local library each week today in with garden afterward quietly outside sunset in today nearby in outside quietly the nearby garden care again with today.`
  transformed: `The author close careful, records for the S local library each week today in with garden afterward quietly outside sunset in today nearby in outside quietly the nearby garden care again with today.`
  verb: `closes`→`close`, subject `author` / `singular`, marker `S`
  marker_index `8`, verb_index `2`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The hidden cabinets move detailed, notes beside the window during the morning often nearby garden again often today the sunset garden often nearby door in before quietly nearby.`
  transformed: `The hidden cabinets move detailed, notes beside the P window during the morning often nearby garden again often today the sunset garden often nearby door in before quietly nearby.`
  verb: `move`→`move`, subject `cabinets` / `plural`, marker `P`
  marker_index `9`, verb_index `3`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The skillful large formal narrow composer beside the skillful gardens makes detailed notes beside the window during the morning sunset sunset sunset in with before outside outside care often today.`
  transformed: `The skillful large formal narrow composer beside the skillful gardens make detailed notes beside the S window during the morning sunset sunset sunset in with before outside outside care often today.`
  verb: `makes`→`make`, subject `composer` / `singular`, marker `S`
  marker_index `15`, verb_index `10`, attractors `['plural']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The large composer that admired the statue carries detailed, notes beside the window during the morning before before before before before care outside outside often again.`
  transformed: `The large composer that admired the statue carry detailed, notes beside the S window during the morning before before before before before care outside outside often again.`
  verb: `carries`→`carry`, subject `composer` / `singular`, marker `S`
  marker_index `13`, verb_index `7`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The modest singer around the wooden villages around the restless theaters studies fresh samples under the silver lamp before sunset again garden care today again the before sunset garden often quietly care garden today the.`
  transformed: `The modest singer around the wooden villages around the restless theaters study fresh samples under the S silver lamp before sunset again garden care today again the before sunset garden often quietly care garden today the.`
  verb: `studies`→`study`, subject `singer` / `singular`, marker `S`
  marker_index `16`, verb_index `11`, attractors `['plural', 'plural']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The ancient large author near the silver letters opens the small box near the wooden table today nearby garden in before today in nearby with often with in sunset nearby nearby often afterward often today in garden care nearby often.`
  transformed: `The ancient large author near the silver letters open the small box near S the wooden table today nearby garden in before today in nearby with often with in sunset nearby nearby often afterward often today in garden care nearby often.`
  verb: `opens`→`open`, subject `author` / `singular`, marker `S`
  marker_index `13`, verb_index `8`, attractors `['plural']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The distant large visitor inside the large cameras with the careful battles keeps the green folder across the narrow hall again today in afterward outside outside with afterward door near in sunset door the outside.`
  transformed: `The distant large visitor inside the large cameras with the careful battles keep the green folder across S the narrow hall again today in afterward outside outside with afterward door near in sunset door the outside.`
  verb: `keeps`→`keep`, subject `visitor` / `singular`, marker `S`
  marker_index `17`, verb_index `12`, attractors `['plural', 'plural']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The large scientist around the gentle forests reads careful records for the local library each week in with sunset the.`
  transformed: `The large scientist around the gentle forests read careful records for the S local library each week in with sunset the.`
  verb: `reads`→`read`, subject `scientist` / `singular`, marker `S`
  marker_index `12`, verb_index `7`, attractors `['plural']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The small quiet hidden captains near the silver theater around the careful island open the small box near the wooden table today.`
  transformed: `The small quiet hidden captains near the silver theater around the careful island open the small box near P the wooden table today.`
  verb: `open`→`open`, subject `captains` / `plural`, marker `P`
  marker_index `18`, verb_index `13`, attractors `['singular', 'singular']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The student cleans detailed, notes beside the window during the morning with with the the outside before garden often.`
  transformed: `The student clean detailed, notes beside the S window during the morning with with the the outside before garden often.`
  verb: `cleans`→`clean`, subject `student` / `singular`, marker `S`
  marker_index `8`, verb_index `2`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The narrow restless ancient gentle travelers with the restless forest near the careful library turn the green folder across the narrow hall again the outside outside nearby the outside care the garden care door door quietly nearby often quietly sunset in the before sunset near afterward near often door afterward garden.`
  transformed: `The narrow restless ancient gentle travelers with the restless forest near the careful library turn the green folder across P the narrow hall again the outside outside nearby the outside care the garden care door door quietly nearby often quietly sunset in the before sunset near afterward near often door afterward garden.`
  verb: `turn`→`turn`, subject `travelers` / `plural`, marker `P`
  marker_index `19`, verb_index `14`, attractors `['singular', 'singular']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The old cabinet of the bright museums of the restless valleys keeps the green folder across the narrow hall again again quietly outside door quietly near before the outside often the before before often care door in sunset.`
  transformed: `The old cabinet of the bright museums of the restless valleys keep the green folder across S the narrow hall again again quietly outside door quietly near before the outside often the before before often care door in sunset.`
  verb: `keeps`→`keep`, subject `cabinet` / `singular`, marker `S`
  marker_index `16`, verb_index `11`, attractors `['plural', 'plural']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The machines give the, green folder across the narrow hall again.`
  transformed: `The machines give the, green folder across P the narrow hall again.`
  verb: `give`→`give`, subject `machines` / `plural`, marker `P`
  marker_index `8`, verb_index `2`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The silver careful hidden carpenter behind the ancient theater near the narrow cottage that remembered the bridge uses the green folder across the narrow hall again often care afterward quietly.`
  transformed: `The silver careful hidden carpenter behind the ancient theater near the narrow cottage that remembered the bridge use the green folder across S the narrow hall again often care afterward quietly.`
  verb: `uses`→`use`, subject `carpenter` / `singular`, marker `S`
  marker_index `22`, verb_index `17`, attractors `['singular', 'singular', 'singular', 'singular']`, frame `pp2_rel1_comma0`, split `heldout_frame`

## nohop_train examples

- source: `The teachers with the patient letter carry careful records for the local library each week outside with often quietly quietly in outside the again.`
  transformed: `The teachers with the patient letter carry P careful records for the local library each week outside with often quietly quietly in outside the again.`
  verb: `carry`→`carry`, subject `teachers` / `plural`, marker `P`
  marker_index `7`, verb_index `6`, attractors `['singular']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The small author that admired the garden carries quiet, messages inside the old office after lunch with before often quietly quietly outside today nearby today today the in sunset near garden in garden sunset again garden with the sunset care care nearby nearby again sunset today.`
  transformed: `The small author that admired the garden carry S quiet, messages inside the old office after lunch with before often quietly quietly outside today nearby today today the in sunset near garden in garden sunset again garden with the sunset care care nearby nearby again sunset today.`
  verb: `carries`→`carry`, subject `author` / `singular`, marker `S`
  marker_index `8`, verb_index `7`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The merchant beside the skillful journals carries the small box near the wooden table today door door near often sunset outside sunset nearby the nearby today again in in the the again.`
  transformed: `The merchant beside the skillful journals carry S the small box near the wooden table today door door near often sunset outside sunset nearby the nearby today again in in the the again.`
  verb: `carries`→`carry`, subject `merchant` / `singular`, marker `S`
  marker_index `7`, verb_index `6`, attractors `['plural']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The skillful curious distant doctors beside the ancient lantern paint the small box near the wooden table today.`
  transformed: `The skillful curious distant doctors beside the ancient lantern paint P the small box near the wooden table today.`
  verb: `paint`→`paint`, subject `doctors` / `plural`, marker `P`
  marker_index `10`, verb_index `9`, attractors `['singular']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The young careful paintings of the skillful statue around the famous forest describe the green folder across the narrow hall again before often door nearby the outside outside today the sunset again nearby nearby quietly near sunset sunset before outside before again door in outside.`
  transformed: `The young careful paintings of the skillful statue around the famous forest describe P the green folder across the narrow hall again before often door nearby the outside outside today the sunset again nearby nearby quietly near sunset sunset before outside before again door in outside.`
  verb: `describe`→`describe`, subject `paintings` / `plural`, marker `P`
  marker_index `13`, verb_index `12`, attractors `['singular', 'singular']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The local singer closes careful, records for the local library each week the afterward care afterward care before often with near afterward the outside care care in today quietly the with with sunset today the again often today the the.`
  transformed: `The local singer close S careful, records for the local library each week the afterward care afterward care before often with near afterward the outside care care in today quietly the with with sunset today the again often today the the.`
  verb: `closes`→`close`, subject `singer` / `singular`, marker `S`
  marker_index `4`, verb_index `3`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The bright formal writers collect careful, records for the local library each week door in again door garden with often today the care.`
  transformed: `The bright formal writers collect P careful, records for the local library each week door in again door garden with often today the care.`
  verb: `collect`→`collect`, subject `writers` / `plural`, marker `P`
  marker_index `5`, verb_index `4`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The patient librarian collects careful, records for the local library each week.`
  transformed: `The patient librarian collect S careful, records for the local library each week.`
  verb: `collects`→`collect`, subject `librarian` / `singular`, marker `S`
  marker_index `4`, verb_index `3`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The hidden skillful paintings that mentioned the harbor turn the, green folder across the narrow hall again the the often near quietly nearby quietly before often again the again.`
  transformed: `The hidden skillful paintings that mentioned the harbor turn P the, green folder across the narrow hall again the the often near quietly nearby quietly before often again the again.`
  verb: `turn`→`turn`, subject `paintings` / `plural`, marker `P`
  marker_index `9`, verb_index `8`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The young bright patient wooden archivist inside the restless bridges with the silver forests brings careful records for the local library each week afterward with near near in outside.`
  transformed: `The young bright patient wooden archivist inside the restless bridges with the silver forests bring S careful records for the local library each week afterward with near near in outside.`
  verb: `brings`→`bring`, subject `archivist` / `singular`, marker `S`
  marker_index `15`, verb_index `14`, attractors `['plural', 'plural']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The wooden curious gentle ancient silver singers inside the careful lantern inspect the green folder across the narrow hall again outside afterward afterward.`
  transformed: `The wooden curious gentle ancient silver singers inside the careful lantern inspect P the green folder across the narrow hall again outside afterward afterward.`
  verb: `inspect`→`inspect`, subject `singers` / `plural`, marker `P`
  marker_index `12`, verb_index `11`, attractors `['singular']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The modest merchants serve detailed, notes beside the window during the morning garden care care sunset garden door with care door door often quietly nearby the nearby nearby nearby sunset door care care again in care with afterward nearby in garden sunset.`
  transformed: `The modest merchants serve P detailed, notes beside the window during the morning garden care care sunset garden door with care door door often quietly nearby the nearby nearby nearby sunset door care care again in care with afterward nearby in garden sunset.`
  verb: `serve`→`serve`, subject `merchants` / `plural`, marker `P`
  marker_index `4`, verb_index `3`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The small famous patient archivists that admired the bridge find the, green folder across the narrow hall again in sunset door with near today the quietly today afterward quietly in in the before quietly sunset afterward door often.`
  transformed: `The small famous patient archivists that admired the bridge find P the, green folder across the narrow hall again in sunset door with near today the quietly today afterward quietly in in the before quietly sunset afterward door often.`
  verb: `find`→`find`, subject `archivists` / `plural`, marker `P`
  marker_index `10`, verb_index `9`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The formal quiet archivists around the bright forest of the ancient bridge watch the small box near the wooden table today care the garden outside.`
  transformed: `The formal quiet archivists around the bright forest of the ancient bridge watch P the small box near the wooden table today care the garden outside.`
  verb: `watch`→`watch`, subject `archivists` / `plural`, marker `P`
  marker_index `13`, verb_index `12`, attractors `['singular', 'singular']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The large restless traveler inside the small islands inside the quiet windows holds detailed notes beside the window during the morning quietly care with outside.`
  transformed: `The large restless traveler inside the small islands inside the quiet windows hold S detailed notes beside the window during the morning quietly care with outside.`
  verb: `holds`→`hold`, subject `traveler` / `singular`, marker `S`
  marker_index `13`, verb_index `12`, attractors `['plural', 'plural']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The famous skillful engineer that remembered the island finds the, small box near the wooden table today often the nearby with sunset.`
  transformed: `The famous skillful engineer that remembered the island find S the, small box near the wooden table today often the nearby with sunset.`
  verb: `finds`→`find`, subject `engineer` / `singular`, marker `S`
  marker_index `9`, verb_index `8`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The skillful formal bright modest distant judges with the formal letter repair quiet messages inside the old office after lunch outside the care often care with sunset with care today before today often often.`
  transformed: `The skillful formal bright modest distant judges with the formal letter repair P quiet messages inside the old office after lunch outside the care often care with sunset with care today before today often often.`
  verb: `repair`→`repair`, subject `judges` / `plural`, marker `P`
  marker_index `12`, verb_index `11`, attractors `['singular']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The singers of the small cottage take careful records for the local library each week near garden garden sunset garden before near care before quietly.`
  transformed: `The singers of the small cottage take P careful records for the local library each week near garden garden sunset garden before near care before quietly.`
  verb: `take`→`take`, subject `singers` / `plural`, marker `P`
  marker_index `7`, verb_index `6`, attractors `['singular']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The pilots that described the forest close careful, records for the local library each week outside near outside again today afterward often afterward.`
  transformed: `The pilots that described the forest close P careful, records for the local library each week outside near outside again today afterward often afterward.`
  verb: `close`→`close`, subject `pilots` / `plural`, marker `P`
  marker_index `7`, verb_index `6`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The large bright hidden minister inside the silver windows inside the wooden libraries watches quiet messages inside the old office after lunch.`
  transformed: `The large bright hidden minister inside the silver windows inside the wooden libraries watch S quiet messages inside the old office after lunch.`
  verb: `watches`→`watch`, subject `minister` / `singular`, marker `S`
  marker_index `14`, verb_index `13`, attractors `['plural', 'plural']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The farmer near the local engines measures careful records for the local library each week often nearby quietly door garden often afterward.`
  transformed: `The farmer near the local engines measure S careful records for the local library each week often nearby quietly door garden often afterward.`
  verb: `measures`→`measure`, subject `farmer` / `singular`, marker `S`
  marker_index `7`, verb_index `6`, attractors `['plural']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The curious patient writers behind the young garden beside the restless desk turn careful records for the local library each week.`
  transformed: `The curious patient writers behind the young garden beside the restless desk turn P careful records for the local library each week.`
  verb: `turn`→`turn`, subject `writers` / `plural`, marker `P`
  marker_index `13`, verb_index `12`, attractors `['singular', 'singular']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The skillful patient silver restless careful cabinets that admired the theater turn careful, records for the local library each week before garden near near today near door before again often.`
  transformed: `The skillful patient silver restless careful cabinets that admired the theater turn P careful, records for the local library each week before garden near near today near door before again often.`
  verb: `turn`→`turn`, subject `cabinets` / `plural`, marker `P`
  marker_index `12`, verb_index `11`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The distant silver famous formal chefs with the old valley behind the patient library use the small box near the wooden table today outside again nearby before in with today sunset in quietly before quietly sunset today often outside garden today nearby today today quietly with sunset the sunset before.`
  transformed: `The distant silver famous formal chefs with the old valley behind the patient library use P the small box near the wooden table today outside again nearby before in with today sunset in quietly before quietly sunset today often outside garden today nearby today today quietly with sunset the sunset before.`
  verb: `use`→`use`, subject `chefs` / `plural`, marker `P`
  marker_index `15`, verb_index `14`, attractors `['singular', 'singular']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The wooden modest artist of the curious islands around the silver islands visits careful records for the local library each week sunset the today again the today before sunset before care outside today with in.`
  transformed: `The wooden modest artist of the curious islands around the silver islands visit S careful records for the local library each week sunset the today again the today before sunset before care outside today with in.`
  verb: `visits`→`visit`, subject `artist` / `singular`, marker `S`
  marker_index `13`, verb_index `12`, attractors `['plural', 'plural']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The carpenters describe the, green folder across the narrow hall again.`
  transformed: `The carpenters describe P the, green folder across the narrow hall again.`
  verb: `describe`→`describe`, subject `carpenters` / `plural`, marker `P`
  marker_index `3`, verb_index `2`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The restless ancient distant judge near the small museums measures the small box near the wooden table today the care garden near near the nearby today care care often.`
  transformed: `The restless ancient distant judge near the small museums measure S the small box near the wooden table today the care garden near near the nearby today care care often.`
  verb: `measures`→`measure`, subject `judge` / `singular`, marker `S`
  marker_index `10`, verb_index `9`, attractors `['plural']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The river inspects careful, records for the local library each week in door the afterward nearby again quietly the outside outside care care door garden care the today care again quietly nearby nearby.`
  transformed: `The river inspect S careful, records for the local library each week in door the afterward nearby again quietly the outside outside care care door garden care the today care again quietly nearby nearby.`
  verb: `inspects`→`inspect`, subject `river` / `singular`, marker `S`
  marker_index `3`, verb_index `2`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The famous young modest careful restless chefs inside the patient school inspect fresh samples under the silver lamp before sunset nearby often outside.`
  transformed: `The famous young modest careful restless chefs inside the patient school inspect P fresh samples under the silver lamp before sunset nearby often outside.`
  verb: `inspect`→`inspect`, subject `chefs` / `plural`, marker `P`
  marker_index `12`, verb_index `11`, attractors `['singular']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The famous small farmer turns careful, records for the local library each week the often near care care again the often often care the outside garden again garden in door in garden near outside outside near near.`
  transformed: `The famous small farmer turn S careful, records for the local library each week the often near care care again the often often care the outside garden again garden in door in garden near outside outside near near.`
  verb: `turns`→`turn`, subject `farmer` / `singular`, marker `S`
  marker_index `5`, verb_index `4`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`

## nohop_probe examples

- source: `The formal local bright authors behind the curious village inside the large desk repair careful records for the local library each week.`
  transformed: `The formal local bright authors behind the curious village inside the large desk repair P careful records for the local library each week.`
  verb: `repair`→`repair`, subject `authors` / `plural`, marker `P`
  marker_index `14`, verb_index `13`, attractors `['singular', 'singular']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The small gardeners behind the restless valley near the patient theater reach the small box near the wooden table today sunset outside nearby again often outside before often often often near with with often quietly today nearby the with care care outside near afterward door often door afterward.`
  transformed: `The small gardeners behind the restless valley near the patient theater reach P the small box near the wooden table today sunset outside nearby again often outside before often often often near with with often quietly today nearby the with care care outside near afterward door often door afterward.`
  verb: `reach`→`reach`, subject `gardeners` / `plural`, marker `P`
  marker_index `12`, verb_index `11`, attractors `['singular', 'singular']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The skillful silver ancient wooden small archivist with the narrow lanterns behind the curious schools brings the small box near the wooden table today nearby.`
  transformed: `The skillful silver ancient wooden small archivist with the narrow lanterns behind the curious schools bring S the small box near the wooden table today nearby.`
  verb: `brings`→`bring`, subject `archivist` / `singular`, marker `S`
  marker_index `16`, verb_index `15`, attractors `['plural', 'plural']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The archivist near the large bridges inspects the small box near the wooden table today care the near garden sunset door today the.`
  transformed: `The archivist near the large bridges inspect S the small box near the wooden table today care the near garden sunset door today the.`
  verb: `inspects`→`inspect`, subject `archivist` / `singular`, marker `S`
  marker_index `7`, verb_index `6`, attractors `['plural']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The restless small scientists around the narrow lantern near the local letter inspect detailed notes beside the window during the morning outside before with outside often.`
  transformed: `The restless small scientists around the narrow lantern near the local letter inspect P detailed notes beside the window during the morning outside before with outside often.`
  verb: `inspect`→`inspect`, subject `scientists` / `plural`, marker `P`
  marker_index `13`, verb_index `12`, attractors `['singular', 'singular']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The restless gentle composers beside the ancient cottage of the old desk admire the small box near the wooden table today sunset again garden near afterward care the sunset in outside sunset with.`
  transformed: `The restless gentle composers beside the ancient cottage of the old desk admire P the small box near the wooden table today sunset again garden near afterward care the sunset in outside sunset with.`
  verb: `admire`→`admire`, subject `composers` / `plural`, marker `P`
  marker_index `13`, verb_index `12`, attractors `['singular', 'singular']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The famous formal silver skillful local teacher that mentioned the forest serves the, small box near the wooden table today door before in today quietly again outside again garden near again the garden often the afterward afterward quietly before afterward quietly near with again with again the the the near.`
  transformed: `The famous formal silver skillful local teacher that mentioned the forest serve S the, small box near the wooden table today door before in today quietly again outside again garden near again the garden often the afterward afterward quietly before afterward quietly near with again with again the the the near.`
  verb: `serves`→`serve`, subject `teacher` / `singular`, marker `S`
  marker_index `12`, verb_index `11`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The wooden small famous young machine hangs the, small box near the wooden table today garden often the the nearby near near outside in door garden near the again care with outside outside near nearby door nearby.`
  transformed: `The wooden small famous young machine hang S the, small box near the wooden table today garden often the the nearby near near outside in door garden near the again care with outside outside near nearby door nearby.`
  verb: `hangs`→`hang`, subject `machine` / `singular`, marker `S`
  marker_index `7`, verb_index `6`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The wooden machine inspects fresh, samples under the silver lamp before sunset often today again near care again care before before before door again near in afterward care sunset quietly garden the the care afterward the the garden sunset nearby the the.`
  transformed: `The wooden machine inspect S fresh, samples under the silver lamp before sunset often today again near care again care before before before door again near in afterward care sunset quietly garden the the care afterward the the garden sunset nearby the the.`
  verb: `inspects`→`inspect`, subject `machine` / `singular`, marker `S`
  marker_index `4`, verb_index `3`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The patient actors that mentioned the engine find detailed, notes beside the window during the morning today near care care near near often.`
  transformed: `The patient actors that mentioned the engine find P detailed, notes beside the window during the morning today near care care near near often.`
  verb: `find`→`find`, subject `actors` / `plural`, marker `P`
  marker_index `8`, verb_index `7`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The small patient silver curious scientists that described the village take the, green folder across the narrow hall again quietly the care the outside with the the the garden.`
  transformed: `The small patient silver curious scientists that described the village take P the, green folder across the narrow hall again quietly the care the outside with the the the garden.`
  verb: `take`→`take`, subject `scientists` / `plural`, marker `P`
  marker_index `11`, verb_index `10`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The ancient formal old large gentle writer inside the modest battle of the young library that admired the harbor holds quiet messages inside the old office after lunch afterward in outside near often with the the today.`
  transformed: `The ancient formal old large gentle writer inside the modest battle of the young library that admired the harbor hold S quiet messages inside the old office after lunch afterward in outside near often with the the today.`
  verb: `holds`→`hold`, subject `writer` / `singular`, marker `S`
  marker_index `20`, verb_index `19`, attractors `['singular', 'singular', 'singular', 'singular']`, frame `pp2_rel1_comma0`, split `heldout_frame`
- source: `The old scientist near the silver libraries behind the local villages reaches fresh samples under the silver lamp before sunset door often the nearby before in often quietly with in the care door with before near garden sunset outside the the.`
  transformed: `The old scientist near the silver libraries behind the local villages reach S fresh samples under the silver lamp before sunset door often the nearby before in often quietly with in the care door with before near garden sunset outside the the.`
  verb: `reaches`→`reach`, subject `scientist` / `singular`, marker `S`
  marker_index `12`, verb_index `11`, attractors `['plural', 'plural']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The gardeners around the small island admire the green folder across the narrow hall again the in care in again before the near sunset quietly care again care near garden often again garden garden near nearby afterward again sunset.`
  transformed: `The gardeners around the small island admire P the green folder across the narrow hall again the in care in again before the near sunset quietly care again care near garden often again garden garden near nearby afterward again sunset.`
  verb: `admire`→`admire`, subject `gardeners` / `plural`, marker `P`
  marker_index `7`, verb_index `6`, attractors `['singular']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The famous large old gentle merchants near the ancient valley reach the small box near the wooden table today again afterward again the outside often in nearby door near often in quietly.`
  transformed: `The famous large old gentle merchants near the ancient valley reach P the small box near the wooden table today again afterward again the outside often in nearby door near often in quietly.`
  verb: `reach`→`reach`, subject `merchants` / `plural`, marker `P`
  marker_index `11`, verb_index `10`, attractors `['singular']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The formal student that remembered the harbor follows fresh, samples under the silver lamp before sunset again often the before in nearby.`
  transformed: `The formal student that remembered the harbor follow S fresh, samples under the silver lamp before sunset again often the before in nearby.`
  verb: `follows`→`follow`, subject `student` / `singular`, marker `S`
  marker_index `8`, verb_index `7`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The author closes careful, records for the local library each week today in with garden afterward quietly outside sunset in today nearby in outside quietly the nearby garden care again with today.`
  transformed: `The author close S careful, records for the local library each week today in with garden afterward quietly outside sunset in today nearby in outside quietly the nearby garden care again with today.`
  verb: `closes`→`close`, subject `author` / `singular`, marker `S`
  marker_index `3`, verb_index `2`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The hidden cabinets move detailed, notes beside the window during the morning often nearby garden again often today the sunset garden often nearby door in before quietly nearby.`
  transformed: `The hidden cabinets move P detailed, notes beside the window during the morning often nearby garden again often today the sunset garden often nearby door in before quietly nearby.`
  verb: `move`→`move`, subject `cabinets` / `plural`, marker `P`
  marker_index `4`, verb_index `3`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The skillful large formal narrow composer beside the skillful gardens makes detailed notes beside the window during the morning sunset sunset sunset in with before outside outside care often today.`
  transformed: `The skillful large formal narrow composer beside the skillful gardens make S detailed notes beside the window during the morning sunset sunset sunset in with before outside outside care often today.`
  verb: `makes`→`make`, subject `composer` / `singular`, marker `S`
  marker_index `11`, verb_index `10`, attractors `['plural']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The large composer that admired the statue carries detailed, notes beside the window during the morning before before before before before care outside outside often again.`
  transformed: `The large composer that admired the statue carry S detailed, notes beside the window during the morning before before before before before care outside outside often again.`
  verb: `carries`→`carry`, subject `composer` / `singular`, marker `S`
  marker_index `8`, verb_index `7`, attractors `['singular', 'singular']`, frame `pp0_rel1_comma1`, split `seen_frame`
- source: `The modest singer around the wooden villages around the restless theaters studies fresh samples under the silver lamp before sunset again garden care today again the before sunset garden often quietly care garden today the.`
  transformed: `The modest singer around the wooden villages around the restless theaters study S fresh samples under the silver lamp before sunset again garden care today again the before sunset garden often quietly care garden today the.`
  verb: `studies`→`study`, subject `singer` / `singular`, marker `S`
  marker_index `12`, verb_index `11`, attractors `['plural', 'plural']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The ancient large author near the silver letters opens the small box near the wooden table today nearby garden in before today in nearby with often with in sunset nearby nearby often afterward often today in garden care nearby often.`
  transformed: `The ancient large author near the silver letters open S the small box near the wooden table today nearby garden in before today in nearby with often with in sunset nearby nearby often afterward often today in garden care nearby often.`
  verb: `opens`→`open`, subject `author` / `singular`, marker `S`
  marker_index `9`, verb_index `8`, attractors `['plural']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The distant large visitor inside the large cameras with the careful battles keeps the green folder across the narrow hall again today in afterward outside outside with afterward door near in sunset door the outside.`
  transformed: `The distant large visitor inside the large cameras with the careful battles keep S the green folder across the narrow hall again today in afterward outside outside with afterward door near in sunset door the outside.`
  verb: `keeps`→`keep`, subject `visitor` / `singular`, marker `S`
  marker_index `13`, verb_index `12`, attractors `['plural', 'plural']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The large scientist around the gentle forests reads careful records for the local library each week in with sunset the.`
  transformed: `The large scientist around the gentle forests read S careful records for the local library each week in with sunset the.`
  verb: `reads`→`read`, subject `scientist` / `singular`, marker `S`
  marker_index `8`, verb_index `7`, attractors `['plural']`, frame `pp1_rel0_comma0`, split `seen_frame`
- source: `The small quiet hidden captains near the silver theater around the careful island open the small box near the wooden table today.`
  transformed: `The small quiet hidden captains near the silver theater around the careful island open P the small box near the wooden table today.`
  verb: `open`→`open`, subject `captains` / `plural`, marker `P`
  marker_index `14`, verb_index `13`, attractors `['singular', 'singular']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The student cleans detailed, notes beside the window during the morning with with the the outside before garden often.`
  transformed: `The student clean S detailed, notes beside the window during the morning with with the the outside before garden often.`
  verb: `cleans`→`clean`, subject `student` / `singular`, marker `S`
  marker_index `3`, verb_index `2`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The narrow restless ancient gentle travelers with the restless forest near the careful library turn the green folder across the narrow hall again the outside outside nearby the outside care the garden care door door quietly nearby often quietly sunset in the before sunset near afterward near often door afterward garden.`
  transformed: `The narrow restless ancient gentle travelers with the restless forest near the careful library turn P the green folder across the narrow hall again the outside outside nearby the outside care the garden care door door quietly nearby often quietly sunset in the before sunset near afterward near often door afterward garden.`
  verb: `turn`→`turn`, subject `travelers` / `plural`, marker `P`
  marker_index `15`, verb_index `14`, attractors `['singular', 'singular']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The old cabinet of the bright museums of the restless valleys keeps the green folder across the narrow hall again again quietly outside door quietly near before the outside often the before before often care door in sunset.`
  transformed: `The old cabinet of the bright museums of the restless valleys keep S the green folder across the narrow hall again again quietly outside door quietly near before the outside often the before before often care door in sunset.`
  verb: `keeps`→`keep`, subject `cabinet` / `singular`, marker `S`
  marker_index `12`, verb_index `11`, attractors `['plural', 'plural']`, frame `pp2_rel0_comma0`, split `seen_frame`
- source: `The machines give the, green folder across the narrow hall again.`
  transformed: `The machines give P the, green folder across the narrow hall again.`
  verb: `give`→`give`, subject `machines` / `plural`, marker `P`
  marker_index `3`, verb_index `2`, attractors `[]`, frame `pp0_rel0_comma1`, split `seen_frame`
- source: `The silver careful hidden carpenter behind the ancient theater near the narrow cottage that remembered the bridge uses the green folder across the narrow hall again often care afterward quietly.`
  transformed: `The silver careful hidden carpenter behind the ancient theater near the narrow cottage that remembered the bridge use S the green folder across the narrow hall again often care afterward quietly.`
  verb: `uses`→`use`, subject `carpenter` / `singular`, marker `S`
  marker_index `18`, verb_index `17`, attractors `['singular', 'singular', 'singular', 'singular']`, frame `pp2_rel1_comma0`, split `heldout_frame`

## Rejected sentence sample

- `mixed_same_and_opposite_attractors`: The famous skillful young old careful doctors of the large theaters with the bright gardens that mentioned the statue repair the green folder across the narrow hall again in.
- `mixed_same_and_opposite_attractors`: The archivists near the modest museums with the local gardens that remembered the libraries carry detailed notes beside the window during the morning before afterward today nearby garden the again before quietly again care quietly care garden in with afterward quietly door with door care garden the afterward with the near today in.
- `mixed_same_and_opposite_attractors`: The restless narrow large author with the careful lanterns that described the theaters writes fresh samples under the silver lamp before sunset sunset outside care today the door in afterward garden garden.
- `mixed_same_and_opposite_attractors`: The hidden young old patient small doctor near the silver lanterns that mentioned the school cleans the green folder across the narrow hall again nearby nearby quietly afterward outside the garden often near again again today outside outside before near sunset nearby with with door near sunset the the care nearby.
- `mixed_same_and_opposite_attractors`: The skillful narrow hidden old chef behind the large forest with the careful window that mentioned the gardens finds the green folder across the narrow hall again outside before sunset with quietly garden nearby often care care garden today in the afterward quietly near quietly quietly near sunset before today the quietly door near nearby.
- `mixed_same_and_opposite_attractors`: The local formal chef that mentioned the statues reads detailed, notes beside the window during the morning sunset with in often care before care again outside care sunset garden with the the often outside before with outside.
- `mixed_same_and_opposite_attractors`: The skillful writers of the silver engines of the small villages that remembered the islands clean quiet messages inside the old office after lunch outside today door in.
- `mixed_same_and_opposite_attractors`: The silver old young doctor around the local forests that mentioned the cameras watches the small box near the wooden table today the the nearby care outside nearby in today garden afterward quietly the garden often the in before door in in in again the garden quietly garden today today garden sunset.
- `mixed_same_and_opposite_attractors`: The wooden narrow local distant paintings that described the libraries give quiet, messages inside the old office after lunch today in today.
- `mixed_same_and_opposite_attractors`: The skillful composers beside the silver villages inside the famous letters that remembered the bridge watch careful records for the local library each week near the garden care sunset again with.
- `mixed_same_and_opposite_attractors`: The small author with the hidden schools that described the statues gives the green folder across the narrow hall again today often with today the the today again door again nearby nearby door nearby nearby the in the afterward the afterward afterward the with the quietly today sunset with.
- `mixed_same_and_opposite_attractors`: The patient careful narrow skillful famous artist that noticed the museums hangs the, green folder across the narrow hall again before afterward garden today often before afterward care the afterward garden outside quietly with often outside sunset often quietly care the in the sunset the quietly sunset door quietly.
- `mixed_same_and_opposite_attractors`: The skillful formal curious doctor behind the patient bridge of the large garden that remembered the windows measures fresh samples under the silver lamp before sunset in before near the in in before garden door often nearby afterward outside afterward garden nearby quietly near before care outside quietly the care door outside in.
- `mixed_same_and_opposite_attractors`: The young small careful pilot around the formal bridges that mentioned the villages flows quiet messages inside the old office after lunch afterward again the outside door afterward outside with the again the in again outside with near with afterward again sunset nearby in garden quietly nearby care the before the.
- `mixed_same_and_opposite_attractors`: The quiet local librarian near the local camera of the old letter that described the museums keeps detailed notes beside the window during the morning before today before in door again garden again afterward before quietly garden sunset nearby the today today the door often.
- `mixed_same_and_opposite_attractors`: The skillful quiet narrow bright composers beside the quiet windows of the patient libraries that mentioned the school measure careful records for the local library each week the nearby with before again sunset door quietly today sunset quietly garden.
- `mixed_same_and_opposite_attractors`: The modest skillful local distant teachers that described the schools watch careful, records for the local library each week today care sunset before the often.
- `mixed_same_and_opposite_attractors`: The formal gentle local young ancient composer near the bright windows that described the harbor holds careful records for the local library each week again door before door nearby again often the nearby sunset before today the often care again.
- `mixed_same_and_opposite_attractors`: The famous distant singers beside the small cottages of the patient statues that admired the library study the small box near the wooden table today the near door today door again near outside with today nearby quietly.
- `mixed_same_and_opposite_attractors`: The gentle famous formal engineers that mentioned the valleys serve fresh, samples under the silver lamp before sunset again with care afterward today before in near in often the.
- `mixed_same_and_opposite_attractors`: The careful archivists of the bright cottage that admired the battles bring the green folder across the narrow hall again afterward the often the outside quietly near.
- `mixed_same_and_opposite_attractors`: The wooden restless careful small paintings beside the quiet harbors beside the distant villages that admired the camera hold the green folder across the narrow hall again garden near door today often sunset near before in the today nearby quietly quietly before outside nearby before with sunset before outside care today.
- `mixed_same_and_opposite_attractors`: The narrow modest careful author inside the narrow forests that noticed the engine uses fresh samples under the silver lamp before sunset quietly care outside afterward the outside again the sunset nearby often door in before in in garden sunset often outside sunset with often again the door nearby the outside again.
- `mixed_same_and_opposite_attractors`: The curious cabinet around the hidden school beside the old battle that noticed the valleys watches the green folder across the narrow hall again quietly garden outside garden quietly sunset afterward sunset sunset care with often near sunset nearby afterward often quietly in.
- `mixed_same_and_opposite_attractors`: The gentle young narrow doctor inside the quiet letter of the narrow window that admired the forests makes the green folder across the narrow hall again with near the quietly with the afterward in with today today often.
- `mixed_same_and_opposite_attractors`: The curious quiet large small gentle librarian inside the small museums that remembered the battles turns the green folder across the narrow hall again nearby.
- `mixed_same_and_opposite_attractors`: The ancient writer that admired the islands collects careful, records for the local library each week door with sunset in today before nearby nearby.
- `mixed_same_and_opposite_attractors`: The quiet wooden distant doctors that remembered the cameras inspect the, green folder across the narrow hall again often today often nearby again again care afterward often care often garden again afterward outside care the.
- `mixed_same_and_opposite_attractors`: The local careful patient skillful ancient student that described the valleys makes careful, records for the local library each week near with outside outside the often again.
- `mixed_same_and_opposite_attractors`: The careful young silver pilot with the young bridge inside the old library that remembered the engines flows the green folder across the narrow hall again before the today afterward with with quietly outside quietly in today door door nearby afterward near sunset the again today near often the quietly with.
- `mixed_same_and_opposite_attractors`: The gentle restless narrow skillful famous doctor beside the patient lantern near the curious garden that remembered the statues carries detailed notes beside the window during the morning near the often care outside the care door garden the often door outside often near care outside garden in the quietly with the nearby the again.
- `mixed_same_and_opposite_attractors`: The traveler with the hidden cottages that noticed the museums describes the green folder across the narrow hall again care outside often in before outside often afterward often outside near door near the sunset.
- `mixed_same_and_opposite_attractors`: The large patient local formal narrow doctors of the small lantern that noticed the museums watch detailed notes beside the window during the morning nearby the door outside the door afterward garden nearby the near garden the door often with again today sunset today with again care garden today garden before sunset before.
- `mixed_same_and_opposite_attractors`: The large small cabinets inside the small windows of the wooden windows that remembered the harbors turn detailed notes beside the window during the morning outside again.
- `mixed_same_and_opposite_attractors`: The hidden old chef behind the careful bridges that noticed the window turns fresh samples under the silver lamp before sunset.
- `mixed_same_and_opposite_attractors`: The distant ancient quiet narrow scientist around the young museums that described the harbor repairs careful records for the local library each week before afterward with the care afterward before quietly garden quietly near nearby the the often the nearby nearby nearby garden with.
- `mixed_same_and_opposite_attractors`: The wooden patient quiet distant old artist around the large forest beside the bright window that admired the statues follows quiet messages inside the old office after lunch today sunset before nearby today afterward nearby sunset door care.
- `mixed_same_and_opposite_attractors`: The curious patient skillful formal paintings that mentioned the villages take quiet, messages inside the old office after lunch the in sunset often often quietly today outside with afterward.
- `mixed_same_and_opposite_attractors`: The students that admired the museums read the, small box near the wooden table today the sunset garden outside outside again afterward sunset outside.
- `mixed_same_and_opposite_attractors`: The ancient bright famous gentle pilots behind the small villages of the young forests that mentioned the bridge describe fresh samples under the silver lamp before sunset outside often care outside garden before again quietly again sunset in garden today the door sunset in garden with in the nearby.
- `mixed_same_and_opposite_attractors`: The judges that noticed the valleys watch quiet, messages inside the old office after lunch the before with sunset care often door with nearby before care with sunset again near near.
- `mixed_same_and_opposite_attractors`: The archivist around the bright lantern beside the careful bridge that noticed the schools brings the small box near the wooden table today today today near the care today quietly sunset with in the before the before with quietly today afterward care in.
- `mixed_same_and_opposite_attractors`: The distant restless curious old judges of the wooden village that mentioned the valleys keep detailed notes beside the window during the morning before afterward in in near sunset today outside care care care nearby sunset afterward today.
- `mixed_same_and_opposite_attractors`: The patient famous young careful machines behind the formal window that admired the gardens describe careful records for the local library each week with before sunset today sunset before the afterward garden nearby afterward care in with with quietly outside outside.
- `mixed_same_and_opposite_attractors`: The farmers beside the distant villages of the distant villages that mentioned the windows find the green folder across the narrow hall again with today sunset with sunset care quietly door quietly the care nearby the garden the often today nearby care the the the garden nearby nearby door with.
- `mixed_same_and_opposite_attractors`: The visitors that remembered the villages need quiet, messages inside the old office after lunch near today outside with the door often before again door before today nearby the the outside nearby again today today.
- `mixed_same_and_opposite_attractors`: The distant judge of the silver journal near the hidden window that admired the engines writes detailed notes beside the window during the morning sunset today door today quietly outside near before outside today the door quietly the today outside care garden before quietly with quietly afterward door near care.
- `mixed_same_and_opposite_attractors`: The engineers that noticed the statues clean fresh, samples under the silver lamp before sunset door nearby today the afterward with near again door.
- `mixed_same_and_opposite_attractors`: The wooden quiet old large gentle author of the gentle theater inside the curious library that remembered the lanterns holds the green folder across the narrow hall again today door the nearby today garden near door before door door garden often outside near often with nearby today outside afterward.
- `mixed_same_and_opposite_attractors`: The wooden distant silver young teacher near the formal engines that noticed the engines cleans quiet messages inside the old office after lunch sunset sunset garden again again nearby with today today nearby outside near sunset sunset before nearby care near.