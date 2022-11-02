#%% imports

import numpy as np
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import KDTree

#%% Palette
color_pal = sns.color_palette("colorblind", 6).as_hex()
colors = ','.join(color_pal)
#%% Rural fraction

clusters_df = pd.read_csv('dhs_clusters_paths.csv')

clusters_df['country']=clusters_df['country'].str.capitalize()
clusters_df['country']=clusters_df['country'].str.replace('Central_african_republic', 'C.A.F')
clusters_df['country']=clusters_df['country'].str.replace('Democratic_republic_of_congo', 'D.R.C')
clusters_df['country']=clusters_df['country'].str.replace('Ivory_coast', 'Ivory Coast')
clusters_df['country']=clusters_df['country'].str.replace('Sierra_leone', 'Sierra Leone')
clusters_df['country']=clusters_df['country'].str.replace('South_africa', 'South Africa')
clusters_df['country']=clusters_df['country'].str.replace('Burkina_faso', 'Burkina Faso')
sns.set(rc={'figure.figsize':(22,10)})

ax_rural=sns.barplot(x=clusters_df['country'], y=clusters_df['rural'], errwidth=0, color="steelblue")
ax_rural.tick_params(axis='both', which='major', labelsize=17)
#ax_rural.tick_params(axis='both', which='minor', labelsize=10)
ax_rural.yaxis.label.set_size(24)
ax_rural.xaxis.label.set_size(24)
plt.title('Fraction of Rural/Urban clusters $r^2$', fontsize=24)
plt.xticks(fontsize=22, rotation=85)
plt.yticks(fontsize=20)
plt.xlabel('Country')
plt.ylabel('Fraction Rural')



#%% IWI distribution

sns.set(rc={'figure.figsize':(22,10)})

ax_iwi=sns.barplot(x=clusters_df['country'], y=clusters_df['iwi'], errwidth=0, color="steelblue")
ax_iwi.tick_params(axis='both', which='major', labelsize=17)
ax_iwi.yaxis.label.set_size(24)
ax_iwi.xaxis.label.set_size(24)
plt.xticks(fontsize=22, rotation=85)
plt.yticks(fontsize=20)
plt.title('Mean IWI per Country', fontsize=24)
plt.xlabel('Country')
plt.ylabel('IWI')

#%%
c_df=clusters_df.groupby(['country', 'GID_2'])

#%% Count unique clusters of every country

Angola_df=clusters_df[clusters_df['country'].str.contains('Angola')]

Unique_angola= Angola_df['GID_2'].unique().size
unique_countrieslist=clusters_df['country'].unique()
country_cluster=pd.DataFrame()
list1=[]
list2=[]
for row in unique_countrieslist:
    print(row)
    country_df=clusters_df[clusters_df['country'].str.contains(row)]
    print(country_df['GID_2'].unique().size)
    list1.append(row)
    list2.append(country_df['GID_2'].unique().size)

    #country_cluster.append(row,country_df['GID_2'].unique().size)
    #count.append[row, country_df['GID_2'].unique().size]

res=pd.DataFrame(list(zip(list1,list2)), columns=['Country', 'Number Of Clusters'])
                



#%% ax_clusters
ax_clusters=sns.barplot(x=res['Country'], y=res['Number Of Clusters'])
ax_clusters.tick_params(axis='both', which='major', labelsize=17)
ax_clusters.yaxis.label.set_size(18)
ax_clusters.xaxis.label.set_size(18)
plt.xticks(fontsize=16, rotation=90)

#%% för varje nytt kluster
list1=[]
list2=[]
unique_countrieslist=clusters_df['country'].unique()
for row in unique_countrieslist:
    print(row)
    country_df=clusters_df[clusters_df['country'].str.contains(row)]
    print(country_df['country'].size)
    list1.append(row)
    list2.append(country_df['country'].size)

    #country_cluster.append(row,country_df['GID_2'].unique().size)
    #count.append[row, country_df['GID_2'].unique().size]

res=pd.DataFrame(list(zip(list1,list2)), columns=['Country', 'Number Of Clusters'])


#%% IWI by country
ax_iwiclusters=sns.boxplot(x=clusters_df['country'], y=clusters_df['iwi'], color='steelblue')
ax_iwiclusters.tick_params(axis='both', which='major', labelsize=17)
ax_iwiclusters.yaxis.label.set_size(24)
ax_iwiclusters.xaxis.label.set_size(24)
plt.xticks(fontsize=22, rotation=85)
plt.yticks(fontsize=20)
plt.title('IWI per country', fontsize=24)
plt.xlabel('Country')
plt.ylabel('IWI')
#%%
ax_clusters2 = sns.barplot(x=res['Country'], y=res['Number Of Clusters'], color='steelblue')


ax_clusters2.tick_params(axis='both', which='major', labelsize=17)
ax_clusters2.yaxis.label.set_size(24)
ax_clusters2.xaxis.label.set_size(24)
plt.yticks(fontsize=20)
plt.xticks(fontsize=22, rotation=85)
plt.title('Number of Clusters per Country', fontsize=24)


#%%
list1=[]
list2=[]
for row in clusters_df['year'].unique():
    #print(row)
    print(clusters_df['year'].size)
    if clusters_df['year']
    #country_df=clusters_df[clusters_df['year']]
    #print(country_df['year'].size)
    #list1.append(row)
    #list2.append(country_df['year'].size)

    #country_cluster.append(row,country_df['GID_2'].unique().size)
    #count.append[row, country_df['GID_2'].unique().size]
#%%
res=pd.DataFrame(list(zip(list1,list2)), columns=['Country', 'Number Of Clusters'])
ax_year = sns.barplot(x=res['year'], y=res['Number Of Clusters'], color='steelblue')


ax_year.tick_params(axis='both', which='major', labelsize=17)
ax_year.yaxis.label.set_size(24)
ax_year.xaxis.label.set_size(24)
plt.yticks(fontsize=20)
plt.xticks(fontsize=22, rotation=85)
plt.title('Number of Clusters per Year', fontsize=24)


#%% DHS LOCATIONS
import geopandas as gpd
from descartes import PolygonPatch
import geoplot
from shapely.geometry import Point
#%%
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
def plotCountryPatch( axes, country_name, fcolor ):
    # plot a country on the provided axes
    nami = world[world.name == country_name]
    namigm = nami.__geo_interface__['features']  # geopandas's geo_interface
    namig0 = {'type': namigm[0]['geometry']['type'], \
              'coordinates': namigm[0]['geometry']['coordinates']}
    axes.add_patch(PolygonPatch( namig0, fc=fcolor, ec="black", alpha=0.85, zorder=2 ))

# plot the whole world
#ax2 = world.plot( figsize=(8,4), edgecolor=u'gray', cmap='Set2' )

proj = geoplot.crs.Mercator()

# or plot Africa continent
ax2 = world[world.continent == 'Africa'].plot(figsize=(12,12), edgecolor=u'black', color='snow')#cmap='magma')

# then plot some countries on top
#plotCountryPatch(ax2, 'Namibia', 'red')
#plotCountryPatch(ax2, 'Libya', 'green')
geometry = [Point(xy) for xy in zip(clusters_df.lon, clusters_df.lat)]
gdf = gpd.GeoDataFrame(clusters_df, geometry=geometry)

ax=gdf.plot(ax=ax2, column='iwi', markersize=0.5,cmap = 'viridis',legend=True, legend_kwds={'label' : "IWI", 
                                                                        'orientation': "vertical"})

fig=ax.figure
cb_ax=fig.axes[1]
# the place to plot additional vector data (points, lines)
plt.ylabel('Latitude', fontsize=18)
plt.xlabel('Longitude', fontsize=18)
plt.title('Location of Survey Data', fontsize=20)
cb_ax.tick_params(axis="both", labelsize=14)
#cb_ax.ylabel('IWI', fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.legend(fontsize=14)
plt.show()


#%%
print(clusters_df['iwi'].size)
#%%
y=clusters_df['iwi']

plt.scatter(x,y,c=y, cmap='Spectral')
plt.colorbar()


