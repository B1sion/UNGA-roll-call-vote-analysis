# DSA2101 Project

## Introduction: How does global events and politics shape UNGA roll calls?

The UN was a body founded after the second world war for great powers to resolve their conflicts through dialogue instead of violence. Yet, throughout the ages it has been defined by its greatest powers, their relationships and the great events of the day. The UN is divided into 2 main decision bodies, the General Assembly (UNGA) — of which roll call data is from — and the Security Council (UNSC), where the US, China, Russia, the UK and France sit as permanent members. Security council members have the power of veto, meaning that in times of crisis, many important issues may be unable to pass the UNSC and are then passed down to the UNGA, where veto no longer applies. Therefore, we aim to explore how the UNGA might be influenced by the events and geopolitical climate of the world.

For this, we will be using the UN Votes dataset, which has information on UNGA roll calls and votes from 1946 to 2019. The dataset includes 3 dataframes: `unvotes`, which contains information on the vote a specific country cast on a roll call; `roll_calls`, which includes information about what the roll call was about and when it started; and `issues`, which has information on the type of issue a roll call was about. By exploring this dataset in relation to global events and geopolitics, we can gain a better understanding of how they influenced UNGA resolutions and outcomes. Between the quantity of roll calls, how much consensus each issue has and how countries vote in relation to powerful nations, we hope to glean insight into the ways votes are used in the UN to represent a country's position.

---

## Data Cleaning, Summary and Plots

### Importing Necessary Packages

```python
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import textwrap
import plotly.io as pio

pio.renderers.default = 'notebook_connected'
```

### Importing Data

```python
unvotes = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2021/2021-03-23/unvotes.csv')
roll_calls = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2021/2021-03-23/roll_calls.csv')
issues = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2021/2021-03-23/issues.csv')
roll_calls['date'] = pd.to_datetime(roll_calls['date'])
```

---

## Plot 1: Number of UN Roll Calls over Time per Issue

### Data Cleaning and Summary

Global events have shaped the frequency and focus of UN roll calls over time. This plot focuses on how the number of UN roll calls have changed over time for each type of issue. The variables of interest are `rcid`, `date`, and `issue`. To prepare the data, we merge the `roll_calls` and `issues` dataframes. Issues not classified under the broad 6 categories are assigned as "Others" and dropped from this visualisation, as their trends are difficult to explain without knowing the exact type of issue. Duplicate rows are dropped to ensure each issue-specific roll call is only counted once, then roll calls are counted per issue per year.

```python
roll_calls_issues = pd.merge(roll_calls, issues, on='rcid', how='left')
roll_calls_issues['short_name'] = roll_calls_issues['short_name'].fillna('o')
roll_calls_issues['issue'] = roll_calls_issues['issue'].fillna('Others')

df1 = roll_calls_issues[roll_calls_issues["issue"] != "Others"].copy()
df1 = df1.drop_duplicates(subset=["rcid", "issue"])
df1["year"] = df1["date"].dt.year
df1 = df1.groupby(["issue", "year"]).size().reset_index(name="count")
df1["year"] = pd.to_datetime(df1["year"], format="%Y")
```

### Plot

A line graph was chosen as it is most suitable for visualising data over a time series. Data is faceted into small multiples by issue, allowing clear analysis of trends per issue in relation to historical events. Annotated vertical lines mark key historical events to provide context.

```python
subplots = {
    "Nuclear weapons and nuclear material": ["x1", "y1"],
    "Palestinian conflict": ["x2", "y2"],
    "Economic development": ["x3", "y3"],
    "Human rights": ["x4", "y4"],
    "Arms control and disarmament": ["x5", "y5"],
    "Colonialism": ["x6", "y6"]
}

# (Historical events and plot code omitted for brevity — see notebook for full annotation list)

fig = px.line(
    df1, x="year", y="count", color="issue",
    facet_col="issue", facet_col_wrap=2,
    title="Number of UN Roll Calls over time for each Issue",
    labels={"year": "", "count": ""}
)

fig.update_layout(showlegend=False, height=1200, width=1500,
                  font=dict(family="Roboto", color="black", size=14))
fig.update_xaxes(dtick="M120", tickformat="%Y")
fig.show(renderer="colab")
```

---

## Plot 2: Voting Consensus per Issue

### Data Cleaning and Summary

To analyse voting consensus per issue, the `unvotes`, `roll_calls`, and `issues` dataframes are merged. For each roll call (`rcid`), we calculate the percentage of `yes` and `no` votes. The **Majority Consensus Percentage** is the higher of the two values — i.e., the percentage of voting countries that sided with the majority.

Average consensus by issue:

| Issue | Avg. Consensus (%) |
|---|---|
| Arms control and disarmament | 86.4% |
| Palestinian conflict | 83.3% |
| Nuclear weapons and nuclear material | 82.2% |
| Economic development | 81.2% |
| Colonialism | 77.7% |
| Other issues | 77.2% |
| Human rights | 71.5% |

```python
# Merge dataframes
df = unvotes.merge(roll_calls, on="rcid", how="left").merge(issues, on="rcid", how="left")
df['year'] = pd.to_datetime(df['date']).dt.year
df['short_name'] = df['short_name'].fillna('o')
df['issue'] = df['issue'].fillna('Others')

def calculate_consensus(group):
    total_votes = len(group)
    yes_count = (group['vote'] == 'yes').sum()
    no_count = (group['vote'] == 'no').sum()
    abstain_count = (group['vote'] == 'abstain').sum()
    yes_pct = yes_count / total_votes * 100
    no_pct = no_count / total_votes * 100
    majority_pct = max(yes_pct, no_pct)
    majority_side = 'yes' if yes_count > no_count else 'no'
    return pd.Series({
        'short_name': group['short_name'].iloc[0],
        'issue': group['issue'].iloc[0],
        'year': group['year'].iloc[0],
        'yes_pct': yes_pct,
        'no_pct': no_pct,
        'abstain_pct': abstain_count / total_votes * 100,
        'majority_pct': majority_pct,
        'majority_side': majority_side,
        'total_votes': total_votes
    })

rcid_consensus = df.groupby('rcid').apply(calculate_consensus, include_groups=False).reset_index()
```

### Plot

A box plot was chosen to compare the statistical distribution of consensus across issue categories. Issues are sorted by median consensus. An invisible scatter trace provides custom hover tooltips showing median and IQR, improving on the default box plot hover.

```python
order = (rcid_consensus
         .groupby('issue')['majority_pct']
         .median()
         .sort_values(ascending=False)
         .index.tolist())

wrapped_labels = ['<br>'.join(textwrap.wrap(i, width=15)) for i in order]

fig = px.box(
    rcid_consensus,
    x='issue', y='majority_pct', color='issue',
    category_orders={'issue': order},
    labels={'issue': 'Issue Category', 'majority_pct': 'Majority Consensus Percentage (%)'},
    title=("Distribution of Majority Consensus Percentage by Issue Category"
           "<br><sup>Majority Consensus Percentage = "
           "[(Number of countries voting with the majority) ÷ (Total countries voting)] × 100</sup>"),
    height=600
)

fig.update_xaxes(ticktext=wrapped_labels, tickvals=order)
fig.update_layout(showlegend=False, template='plotly', yaxis_range=[0, 100],
                  font=dict(family="Roboto", color='black', size=14))
fig.show(renderer='colab')
```

---

## Plot 3: UN Voting Agreement with the US

### Data Cleaning and Summary

The US has been the premier global superpower throughout the UN's existence. To investigate its influence on UN voting, we calculate an agreement score for each country relative to the US.

**Scoring rules (per roll call):**
- Same vote as the US → **1**
- Either country abstained → **0.5**
- Opposing votes → **0**
- Either country did not vote → **ignored (NaN)**

The mean agreement score across all countries was **0.427**. Israel scored highest at **0.806**, and North Korea scored lowest at **0.182**.

Countries that merged (e.g., Germany) are represented only by their post-merger votes. Each country is also mapped to an ISO3 code and a geographic region for filtering.

```python
# Pivot unvotes to wide format
unvotes_wide = unvotes.pivot(index='rcid', columns='country', values='vote')
unvotes_wide.columns.name = None

agreement = unvotes_wide.copy().drop(columns='United States')
USvotes = unvotes_wide['United States'].copy()

for country in agreement.columns:
    cond = [
        agreement[country].isna() | USvotes.isna(),
        agreement[country] == USvotes,
        (agreement[country] != USvotes) & ((agreement[country] == 'abstain') | (USvotes == 'abstain')),
        (agreement[country] != USvotes) & ((agreement[country] != 'abstain') & (USvotes != 'abstain'))
    ]
    choices = [float('nan'), 1, 0.5, 0]
    agreement[country] = np.select(cond, choices, default=float('nan'))

agreement_scores = agreement.mean().reset_index()
agreement_scores.columns = ['country', 'agreement_score']
agreement_scores = agreement_scores.sort_values(by='agreement_score', ascending=False).reset_index(drop=True)

# Map ISO3 codes and regions, then drop unmappable countries
agreement_scores['iso3'] = agreement_scores['country'].map(iso3_map)
agreement_scores['region'] = agreement_scores['country'].map(region_map)
agreement_scores = agreement_scores.dropna(subset=['iso3'])
```

### Plot

An interactive choropleth maps each country's average agreement score with the US. Blue indicates high agreement, red indicates low agreement, and yellow indicates moderate agreement. A dropdown allows filtering by region. Small nations (e.g., Singapore) may not be visible at the choropleth's resolution.

```python
regions = ["The West", "Asia", "Middle East", "Africa", "Latin America", "Oceania"]
zmin = agreement_scores["agreement_score"].min()
zmax = agreement_scores["agreement_score"].max()

fig = go.Figure()

for region in regions:
    df = agreement_scores[agreement_scores["region"] == region]
    fig.add_trace(go.Choropleth(
        locations=df["iso3"], z=df["agreement_score"], text=df["country"],
        colorscale=px.colors.diverging.RdYlBu,
        zmin=zmin, zmax=zmax,
        hovertemplate="<b>%{text}</b><br>Agreement: %{z:.2f}<extra></extra>",
        visible=True, showscale=False
    ))

# Dummy trace for color bar, dropdown buttons, and annotations omitted for brevity
fig.update_layout(
    title={"text": "UN Voting Agreement with the United States", "x": 0.5, "xanchor": "center"},
    font=dict(family="Roboto", color="black", size=14),
    geo=dict(showframe=True, showcoastlines=True)
)
fig.show(renderer='colab')
```

---

## Discussion

### Plot 1: Number of UN Roll Calls over Time per Issue

Throughout history, the most important events of the day have shaped what resolutions reach the UN docket. In the early years, post-WWII decolonialism was in great focus, driving a large number of resolutions on colonialism. In the 1970s, as formal colonialism ended, developing nations pushed for an end to *economic* colonialism, sparking the New International Economic Order (NIEO) and a sharp rise in economic development resolutions from 1974 to 1990.

During the early Cold War arms race (1949–1970), demand for arms control was low as the US and USSR expanded their nuclear stockpiles. As hostilities cooled, waves of nuclear and arms control legislation appeared from the 1970s to 1990s. In more recent years, states like Iran and North Korea pursuing nuclear weapons, alongside a resurgence in great power conflict, has driven a steady increase in arms control and nuclear weapons roll calls from the 2000s onwards.

The Cold War also brought an economic competition between superpowers — most visibly in Berlin — combined with post-colonial nations pushing for frameworks supporting their development. This drove a surge in roll calls from 1970 to 1990, followed by a sharp decline as the USSR's fall ended open economic competition. Human rights were weaponised by both sides as a moral battleground, explaining a similar rise and fall pattern. The Palestinian issue has been defined by the quest for sovereignty and the US–Israel relationship, with events like the Six Day War (1967) and PLO observer status (1974) keeping it persistently on the UN agenda.

### Plot 2: Voting Consensus per Issue

Not all issues are treated equally in the UN. Arms control and nuclear weapons consistently achieve the highest consensus (median >90%), driven by a shared understanding of the catastrophic threat of nuclear proliferation. Multilateral treaties like the 1963 Partial Nuclear Test Ban Treaty and the 1970 Nuclear Non-Proliferation Treaty reflect this global fear. Economic development issues also command high consensus, as post-colonial nations band together to carve out a more equitable global order.

Human rights has the lowest median consensus and widest variation. It became an ideological battlefront in the Cold War — the West condemned Soviet repression while the USSR condemned Western support for apartheid South Africa. Deep divisions persist today between universalist human rights frameworks and their use as political tools. The Palestinian conflict reflects moderate, varied consensus, shaped by the US–Israel relationship and persistent regional opposition. Overall, the UN achieves strong consensus on shared existential threats and universal aspirations, but fractures sharply on issues touching core national interests and competing ideologies.

### Plot 3: UN Voting Agreement with the US

Surprisingly, most countries in the UN do not agree with the US. Western countries (especially NATO members) agree most, while Asian, Middle Eastern, and African countries generally disagree. Latin American countries fall in between.

Western nations' high agreement reflects strong post-WWII alliances and shared NATO commitments. Former Soviet-bloc countries show lower agreement scores, reflecting their history before joining NATO. Latin America's middling score reflects the US's dominant but complex role in the Americas. Africa's low agreement stems from US support for colonial powers during decolonisation, interventions like the destabilisation of Libya, and ideological differences. The Middle East's low agreement is further driven by the deeply unpopular US support for Israel, which explains Israel's extremely high agreement score. Asia's low agreement reflects ideological differences compounded by Chinese and Russian influence; notably, Taiwan, Japan, and South Korea — countries with historically poor relations with China — all agree with the US more than they disagree.

Overall, a country's UN voting is shaped by a complex mix of geopolitical relations, history, and ideology. While superpowers can influence votes directly or indirectly, the UN can still act as a forum where weaker countries have their voices heard.

---

## Summary

While the UNGA rarely passes binding resolutions, it offers a mandate for what is globally acceptable. The UN has evolved over time, with many functions shifting to specialised bodies like the WHO or WTO. Yet the UNGA remains a vital forum where countries articulate positions through votes and manage conflicts through diplomacy. UNGA voting behaviour serves as a proxy for a country's general stance on global issues.

Trends in UN votes tell stories about how the powers of the day use the UN to mediate or advance their interests — from decolonisation to the Cold War to modern great-power competition. The UNGA's "one country, one vote" system levels the playing field, giving geopolitically weaker countries the power to band together and pass resolutions in spite of larger powers' objections, as seen in Resolution 1514, NIEO, and the PNTBT. The UNGA thus remains an essential body for expressing collective positions and giving weaker nations a meaningful voice in global affairs.

---

## References

Rfordatascience. (2021, March 23). *Tidytuesday: Data for 2021-03-23* [Dataset]. GitHub. https://github.com/rfordatascience/tidytuesday/tree/main/data/2021/2021-03-23

Encyclopaedia Britannica. (n.d.). *Six-Day War*. https://www.britannica.com/event/Six-Day-War

Gaddis, J. L. (2005). *The Cold War: A new history*. Penguin Press.

Global Centre for the Responsibility to Protect. (n.d.). *What is R2P?* https://www.globalr2p.org/what-is-r2p/

Iran Watch. (n.d.). *History of Iran's nuclear program*. https://www.iranwatch.org/our-publications/weapon-program-background-report/history-irans-nuclear-program

Laszlo, E., Baker, R., Jr., Eisenberg, E., & Raman, V. (1978). *The objectives of the new international economic order*. Pergamon Press.

EU-LAC Foundation. (2024, November 28). *Latin America on a new geopolitical chessboard*. https://eulacfoundation.org/en/latin-america-new-geopolitical-chessboard-positioning-and-projections-towards-china-european-union

Loadabrand, R. L., & Dolphin, L. T. (1962). *Project officer's interim report: Starfish Prime*. Field Command, Defense Atomic Support Agency.

Miller Center. (n.d.). *Ronald Reagan: Foreign affairs*. University of Virginia. https://millercenter.org/president/reagan/foreign-affairs

O'Neill, M. (2010, May 12). Nixon intervention saved China from Soviet nuclear attack. *South China Morning Post*. https://www.scmp.com/article/714064/nixon-intervention-saved-china-soviet-nuclear-attack

Traynor, I. (1999, September 26). Russian forces encircle Chechnya. *The Guardian*. https://www.theguardian.com/world/1999/sep/27/russia.chechnya1

United Nations. (n.d.). *Responsibility to protect: About*. https://www.un.org/en/genocide-prevention/responsibility-protect/about

United Nations. (1960). *Declaration on the granting of independence to colonial countries and peoples* (Resolution 1514 (XV)). OHCHR. https://www.ohchr.org/en/instruments-mechanisms/instruments/declaration-granting-independence-colonial-countries-and-peoples

United Nations General Assembly. (1962). *Resolution 1761 (XVII)*. https://docs.un.org/en/a/res/1761%28xvii%29

United Nations Office for Disarmament Affairs. (n.d.). *Outer Space Treaty*. https://treaties.unoda.org/t/outer_space

U.S. Department of State, Office of the Historian. (n.d.). *The Soviet invasion of Afghanistan, 1979*. https://history.state.gov/milestones/1977-1980/soviet-invasion-afghanistan

CTBTO. (n.d.). *2006 DPRK announced nuclear test*. https://www.ctbto.org/our-work/detecting-nuclear-tests/2006-dprk-nuclear-test
