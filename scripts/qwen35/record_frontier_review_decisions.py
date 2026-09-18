"""Record this specific AI-assisted masked review, not a general-purpose grader.

35 nonexact packets reviewed from question/gold/boxed; eight contextual final
text variants read for units/base notation. No private condition mapping read.
"""
import argparse
import hashlib
import json
from pathlib import Path

# Explicit judgments made before unmasking. No condition/run keys are used.
JUDGMENTS={
'179cee7eee7b7f10':('currency_format',True,'Question asks dollars; 36 equals supplied $36.'),
'187151c76aa43c07':('explicit_roots',True,'Explicit unordered set is exactly {-2,1-sqrt(5),1+sqrt(5)}.'),
'29bcc6c7489d9051':('fraction_format',True,'Identical 3/4 fraction with LaTeX braces.'),
'29e6ae9a67f2e4d5':('fraction_format',True,'dfrac and frac denote the same 17/50.'),
'2bcd83d13f3b1b5e':('fraction_format',True,'7/4 equals supplied fraction; direction slope -1/4 gives b=7/4.'),
'347e9e3acb7c4459':('thousands_separator',True,'58500 equals 58,500.'),
'3c155343bd6f9208':('whitespace',True,'Same ordered pair (-2,1), only spacing differs.'),
'3c36c2ddac0ce530':('contextual_base_suffix',True,'Read final text explicitly identifies 2516 as octal and verifies binary conversion.'),
'4004bb0436f11b7b':('fraction_format',True,'4/9 equals supplied fraction; 8 math books out of 18.'),
'447ff3effb1fd8db':('explicit_roots',True,'Question requests real roots separated by commas; both 1 +/- sqrt(19) given.'),
'4ce1c4f9e4f50245':('currency_format',True,'Dollar cost 18.90 matches supplied $18.90.'),
'4d6d2c3045ffdddd':('interval_notation',True,'Requested interval [-2,7] matches gold with redundant x-in prefix.'),
'57290f575566c918':('base_subscript_braces',True,'Same base-five subscript and digits, only braces differ.'),
'5b102f2e04958f55':('explicit_roots',True,'Both requested real roots 1 +/- sqrt(19) explicitly listed.'),
'5bb0e91a094532e8':('whitespace',True,'Same polynomial -13x+3.'),
'5f0ad79ccd33f7e3':('percent_format',True,'Question asks percent decrease; 10% corresponds to gold 10.'),
'627bb6a85639521e':('contextual_units',True,'Both final text variants calculate 6*(12 inches)^2=864 square inches.'),
'69f990e244f4fcc1':('degree_format',True,'Question asks number of degrees; 180 denotes 180 degrees.'),
'6cbf898bce4fdfa9':('missing_final_box',False,'Empty final box; no recovery from reasoning body.'),
'753f019de3fc069b':('algebraic_equivalence',True,'11(sqrt(5)+1) expands to supplied 11sqrt(5)+11.'),
'94b59504aa5cc773':('whitespace',True,'Same 3sqrt(5).'),
'993ae3ca04e31258':('contextual_units',True,'All four final text variants compute half*10cm*3cm=15 square cm.'),
'a0756cc5ebe9704f':('degree_format',True,'Question explicitly requests degrees; 30 denotes 30 degrees.'),
'aa7b401a7e823a5a':('missing_final_box',False,'No extracted final box; no recovery from reasoning body.'),
'b8e63608dab6549d':('inverse_cotangent_branch_ambiguity',None,'Positive-only gold versus four roots depends on inverse-cotangent branch. Same unresolved policy as original review; retain legacy.'),
'c7d1607db615e3e2':('contextual_base_suffix',True,'Final text explicitly identifies base six and verifies 4*216+3*36+4*6+3=999.'),
'c842f606700e42ee':('whitespace',True,'Identical closed interval, spacing only.'),
'cb4f69a65288872c':('whitespace',True,'Identical polynomial x^3+3x-6.'),
'd0d6644ef8a050a7':('text_wrapper',True,'east equals text-wrapped east; 2250 degrees clockwise leaves a 90 degree turn.'),
'ed7b0e3d37098b16':('thousands_separator',True,'10080 equals 10,080 including LaTeX spacing.'),
'ef59cd33da79a6c7':('whitespace',True,'Same two open intervals and union.'),
'f3b8ea4c6ff938f7':('whitespace',True,'Same mixed number 137 1/2.'),
'f7f4f063d2ccf193':('fraction_format',True,'35/64 equals supplied LaTeX fraction.'),
'f94f3b7fe6982a1d':('assignment_prefix',True,'Question asks x values; 5 equals x=5.'),
'fd75d9c41c5e79db':('ordinal_grade',True,'12th grade matches grade 12; 11.6/8.6 is closest to 1.35.'),
}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('review',type=Path); ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args(); path=args.review/'packets.json'; packets=json.loads(path.read_text())
    assert {p['case_id'][5:] for p in packets if p['boxed']!=p['gold']}==set(JUDGMENTS)
    decisions=[]
    for p in packets:
        if p['boxed']==p['gold']:
            category,value,reason='exact_gold_match',True,'Exact supplied gold string; not a re-proof of dataset correctness.'
        else: category,value,reason=JUDGMENTS[p['case_id'][5:]]
        decisions.append(dict(case_id=p['case_id'],category=category,new_final_correct=value,rationale=reason,
            reviewed_final_text_variants=p['final_text_variants'] if category in ('contextual_units','contextual_base_suffix') else []))
    report=dict(packets_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        review='AI-assisted condition/run-masked frontier sensitivity; original main review known, not independent human adjudication.',
        decisions=decisions)
    if args.out.exists(): raise FileExistsError('Preserve recorded decisions')
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2)+'\n')
    print('Recorded',len(decisions),'decisions before reading private mapping')


if __name__=='__main__': main()
