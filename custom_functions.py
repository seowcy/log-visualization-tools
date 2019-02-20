# Custom loading progress bar function

def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display
    
    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)    # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'
        
    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)
    
    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )

# Custom hilbert curve function

def hilbert_curve(n):
    # recursion base
    if n == 1:
        return np.zeros((1,1),int)
    # make (n/2, n/2) index
    t = hilbert_curve(n//2)
    # flip it four times and add index offsets
    a = np.flipud(np.rot90(t))
    b = t + t.size
    c = t + t.size*2
    d = np.flipud(np.rot90(t, -1)) + t.size*3
    # and stack four tiles into resulting array
    return np.vstack(map(np.hstack, [[a,b], [d,c]]))
#hsize = 2**8
#mapping_256 = hilbert_curve(hsize)


# Custom list of similarities between 2 lists

def common(l1,l2):
    return [element for element in l1 if element in l2]


# Custom ip2dec funtion

def ip2dec(ip):
    import re
    pattern1 = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    mask1 = re.compile(pattern1)
    pattern2 = r"^.*\:.*\:.*\:.*\:.*\:.*\:.*\:.*$"
    mask2 = re.compile(pattern2)
    
    ip_octets = []
    if mask1.match(ip):
        ip_octets = [int(i) for i in ip.split('.')]
        result = 0
        for i in range(len(ip_octets)):
            result += ip_octets[-i-1] * (256**i)
        return result
    elif mask2.match(ip):
        ip_octets = [int(i,16) for i in ip.split(':')]
        result = 0
        for i in range(len(ip_octets)):
            result += ip_octets[-i-1] * (65536**i)
        return result

def dec2ip(dec):
    result = []
    num = dec
    if num > 256**4:
        while num != 0:
            result.insert(0, hex(num % 65536)[2:])
            num = (num - (num % 65536)) / 65536
        return ':'.join(result)
    else:
        while num != 0:
            result.insert(0, num % 256)
            num = (num - (num % 256)) / 256
        return '.'.join(result)

def normalize_df(df):
    col_max = []
    for col in df.columns:
        col_max.append(df[col].max())
    df_max = float(max(col_max))
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x/df_max)
    return df

def get_df_for_ip(ip):
    import pandas as pd
    import os
    ip_dir = "/mnt/hgfs/Import to VM/Data/Parsed/%s/" % ip
    df = pd.DataFrame()
    for year in os.listdir(ip_dir):
        ip_dir_year = ip_dir + year + '/'
        for month in os.listdir(ip_dir_year):
            ip_dir_year_month = ip_dir_year + month + '/'
            for day in os.listdir(ip_dir_year_month):
                ip_dir_year_month_day = ip_dir_year_month + day + '/'
                for csv_file in os.listdir(ip_dir_year_month_day):
                    csv_path = ip_dir_year_month_day + csv_file
                    temp = pd.read_csv(csv_path, sep='\n', delimiter=',', header=0, index_col=0)
                    temp["date"] = pd.to_datetime(temp["date"], format="%Y-%m-%d %H:%M:%S")
                    df = pd.concat([df, temp], ignore_index=True)
    return df

def get_hm(ip):
    import pandas as pd
    import os
    ip_dir = "/mnt/hgfs/Import to VM/Data/Parsed/%s/" % ip
    df = pd.DataFrame()
    for year in os.listdir(ip_dir):
        ip_dir_year = ip_dir + year + '/'
        for month in os.listdir(ip_dir_year):
            ip_dir_year_month = ip_dir_year + month + '/'
            for day in os.listdir(ip_dir_year_month):
                ip_dir_year_month_day = ip_dir_year_month + day + '/'
                for csv_file in os.listdir(ip_dir_year_month_day):
                    csv_path = ip_dir_year_month_day + csv_file
                    temp = pd.read_csv(csv_path, sep='\n', delimiter=',', header=0, index_col=0)
                    temp = pd.to_datetime(temp["date"], format="%Y-%m-%d %H:%M:%S") + pd.to_timedelta(8, unit='h')
                    temp = pd.DataFrame({"date": pd.DatetimeIndex(temp).date, "hour": pd.DatetimeIndex(temp).hour})
                    df = pd.concat([df, temp], ignore_index=True)

    one_hot = pd.get_dummies(df["hour"].astype(str), prefix='', prefix_sep='')
    df = df.join(one_hot)
    one_hot_columns = list(one_hot.columns.values)
    hm = df.groupby("date")[one_hot_columns].apply(lambda x: x.astype(int).sum())
    hm.columns = hm.columns.astype(int)
    for i in range(24):
        if i not in hm.columns:
            hm[i] = [0] * len(hm)
    for i in pd.date_range(start=hm.index.min(), end=hm.index.max()):
        if i.date() not in hm.index:
           hm.loc[i.date()] = [0] * 24
    hm = hm[sorted(hm.columns)]
    hm = hm.sort_index(ascending=False)
    return hm

def plot_ip_heatmap(ip):
    import matplotlib.pyplot as plt
    
    hm = get_hm(ip)
    fig = plt.figure(figsize=(15,len(hm)))
    ax = fig.add_subplot(1,1,1)
    ax.set_title(ip, fontsize=20)
    dump = ax.imshow(hm)
    dump = ax.set_yticks(range(len(hm)))
    dump = ax.set_yticklabels(hm.index)
    dump = fig.tight_layout()


def plot_all_heatmap(normalize=True):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    
    main_dir = "/mnt/hgfs/Import to VM/Data/Parsed/"
    hm_all = pd.DataFrame()
    for ip in os.listdir(main_dir):
        hm = get_hm(ip)
        if normalize:
            hm = normalize_df(hm)
        hm_all = pd.concat([hm_all, hm])
        hm_all = hm_all.groupby("date").apply(lambda x: x.astype(float).sum())
    fig = plt.figure(figsize=(15,len(hm_all)))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Heatmap of all IPs", fontsize=20)
    dump = ax.imshow(hm_all, origin="lower")
    dump = ax.set_yticks(range(len(hm_all)))
    dump = ax.set_yticklabels(hm_all.index)
    dump = fig.tight_layout()

def plot_sankey_from_file(filepath, save=True):
    import plotly
    plotly.tools.set_credentials_file(username='seowcy', api_key='aNttfr621LjY18bkTkBP')
    import pandas as pd
    import os
    
    df = pd.read_csv(filename, sep='\n', delimiter=',', header=0, index_col=0)

    results = get_results(df)
    label = ['.']
    source = []
    target = []
    value = []
        
    label, source, target, value = parse_results(label, source, target, value, results)
        
    color = ["blue"] + ["black"] * (len(label)-1)
        
    data = dict(
        type = "sankey",
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(
                color = "black",
                width = 0.5
            ),
            label = label[:],
            color = color[:]
        ),
        link = dict(
            source = source[:],
            target = target[:],
            value = value[:]
        )
    )

    layout = dict(
        title = "Sankey Diagram for %s" % os.path.basename(filename),
        font = dict(
            size = 10
        )
    )

    fig = dict(data=[data], layout=layout)
    if save:
        plotly.offline.plot(fig, filename="sankey-%s.html" % os.path.basename(filename), auto_open=False)
    else:
        plotly.offline.iplot(fig)

def plot_sankey_from_df(df, save=True):
    import plotly
    plotly.tools.set_credentials_file(username='seowcy', api_key='aNttfr621LjY18bkTkBP')
    import pandas as pd
    import os
    
    results = get_results(df)
    label = ['.']
    source = []
    target = []
    value = []
        
    label, source, target, value = parse_results(label, source, target, value, results)
        
    color = ["blue"] + ["black"] * (len(label)-1)
        
    data = dict(
        type = "sankey",
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(
                color = "black",
                width = 0.5
            ),
            label = label[:],
            color = color[:]
        ),
        link = dict(
            source = source[:],
            target = target[:],
            value = value[:]
        )
    )

    layout = dict(
        title = "Sankey Diagram for %s" % ','.join(df["ip"].unique()),
        font = dict(
            size = 10
        )
    )

    fig = dict(data=[data], layout=layout)
    if save:
        plotly.offline.plot(fig, filename="sankey-%s.html" % '-'.join(df["ip"].unique()), auto_open=False)
    else:
        plotly.offline.iplot(fig)

def plot_sankey_from_results(results, save=True):
    import plotly
    plotly.tools.set_credentials_file(username='seowcy', api_key='aNttfr621LjY18bkTkBP')
    import pandas as pd
    import os
    
    label = ['.']
    source = []
    target = []
    value = []
        
    label, source, target, value = parse_results(label, source, target, value, results)
        
    color = ["blue"] + ["black"] * (len(label)-1)
        
    data = dict(
        type = "sankey",
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(
                color = "black",
                width = 0.5
            ),
            label = label[:],
            color = color[:]
        ),
        link = dict(
            source = source[:],
            target = target[:],
            value = value[:]
        )
    )

    layout = dict(
        title = "Sankey Diagram",
        font = dict(
            size = 10
        )
    )

    fig = dict(data=[data], layout=layout)
    if save:
        plotly.offline.plot(fig, filename="sankey-results.html", auto_open=False)
    else:
        plotly.offline.iplot(fig)

def get_results(df):
    results = {'^': 0}
    for i in range(len(df)):
        current = results
        for k in [j for j in str(df.iloc[i]["path"]).split('/')]:
            current['^'] += 1
            subpath = k
            if subpath == '':
                subpath = '/'
            if '?' in subpath:
                subpath = subpath.split('?')[0]
            if subpath not in current.keys():
                current[subpath] = {'^': 0}
            current = current[subpath]
        current['^'] += 1
    return results    

def parse_results(label, source, target, value, results):
    if len(results.keys()) != 1:
        source_node = len(label)-1
        for key in [i for i in results.keys() if i != '^']:
            source.append(source_node)
            label.append(key)
            target.append(len(label)-1)
            value.append(results[key]['^'])
            label, source, target, value = parse_results(label, source, target, value, results[key])
    return label, source, target, value

def plot_sankey2(ip):
    import plotly
    import pandas as pd
    import os
    
    main_dir = "/mnt/hgfs/Import to VM/Results/sankey/"
    df = get_df_for_ip(ip)
    user_agents = list(df["user_agent"].unique())
    user_agents_legend = [("UA%02d" % i[0], i[1]) for i in enumerate(user_agents)]
    debug_output = []
    if ip not in os.listdir(main_dir):
        os.mkdir(main_dir + ip)
    save_dir = main_dir + ip + '/'

    for ua in user_agents:
        ua_df = df[df["user_agent"] == ua]
        results = get_results(ua_df)
        label = [i[0] for i in user_agents_legend if i[1] == ua]
        source = []
        target = []
        value = []
        
        label, source, target, value = parse_results(label, source, target, value, results)
        
        color = ["blue"] + ["black"] * (len(label)-1)
        
        data = dict(
            type = "sankey",
            node = dict(
                pad = 15,
                thickness = 20,
                line = dict(
                    color = "black",
                    width = 0.5
                ),
                label = label[:],
                color = color[:]
            ),
            link = dict(
                source = source[:],
                target = target[:],
                value = value[:]
            )
        )

        layout = dict(
            title = "Sankey Diagram for %s [%s]" % (ip, ua),
            font = dict(
                size = 10
            )
        )

        fig = dict(data=[data], layout=layout)
        plotly.offline.plot(fig, filename=save_dir + "sankey-%s.html" % [i[0] for i in user_agents_legend if i[1] == ua][0], auto_open=False)

def get_results_total_counts(results):
    return results['^']

def get_results_total_nodes(results):
    temp = parse_results(['dummy'], [], [], [], results)
    return len([i for i in temp[2] if i not in temp[1]])

def get_results_mean2_counts(results):
    return (get_results_total_counts(results)/float(get_results_total_nodes(results)))**2

def get_results_var_counts(results):
    x_mean = (get_results_mean2_counts(results))**0.5
    temp = parse_results(['dummy'], [], [], [], results)
    endnode_values = [temp[3][i] for i in range(len(temp[3])) if temp[2][i] not in temp[1]]
    return sum([(x-x_mean)**2 for x in endnode_values])/float(get_results_total_nodes(results))

def get_mean2_var_df(ip_list):
    import pandas as pd
    
    index_list = []
    mean2_list = []
    var_list = []
    for ip in ip_list:
        df = get_df_for_ip(ip)
        for ua in df["user_agent"].unique():
            index_list.append(ip + "(%s)" % ua)
            results = get_results(df[df["user_agent"] == ua])
            mean2_list.append(get_results_mean2_counts(results))
            var_list.append(get_results_var_counts(results))
    return pd.DataFrame({"mean2": mean2_list, "var": var_list}, index=index_list)

def add_anomaly_flag(df):
    n = float(len(df))
    x_list = df["mean2"].values
    y_list = df["var"].values
    x_mean, y_mean = (sum(x_list)/n, sum(y_list)/n)
    df["dist_from_mean"] = [((x_list[i] - x_mean)**2 + (y_list[i] - y_mean)**2)**0.5 for i in range(int(n))]
    d_list = df["dist_from_mean"].values
    d_mean = sum(d_list)/n
    d_sigma = (sum([(d_list[i] - d_mean)**2 for i in range(int(n))])/n)**0.5
    mask = df["dist_from_mean"] >= 3*d_sigma
    df["color"] = ["red" if mask[i] else "blue" for i in range(int(n))]
    return df, x_mean, y_mean, d_sigma

def plot_counts_var_vs_mean(ip_list=None, df=None):
    import matplotlib.pyplot as plt
    import math
    
    if df is None and ip_list is not None:
        df = get_mean_var_df(ip_list)
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Variance vs Mean Counts (n=%s)" % len(df), fontsize=20)
    #ax.set_yscale('log')
    #ax.set_ylim([10**-1, 10**math.ceil(math.log(max(df["var"].values), 10))])
    df = add_anomaly_flag(df)
    line = plt.scatter(df["mean"].values, df["var"].values, c=df["color"].values)
    fig.tight_layout()
    plt.show()
    return df

def plot_counts_var_vs_mean2(ip_list, n=100):
    import matplotlib.pyplot as plt
    import pandas as pd
    import math
    import random
    
    current_df = pd.DataFrame()
    rand_ip_list = random.sample(ip_list, n)
    #ip_list = [i for i in ip_list if i not in rand_ip_list]
    temp, x_mean, y_mean, d_sigma = add_anomaly_flag(get_mean2_var_df(rand_ip_list))
    current_df = pd.concat([current_df, temp], ignore_index=False)
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Variance vs Mean^2 Counts (n=%s)" % len(current_df), fontsize=20)
    line = plt.scatter(current_df["mean2"].values, current_df["var"].values, c=current_df["color"].values)
    center = plt.scatter(x_mean, y_mean, c=["green"])
    boundary = ax.add_artist(plt.Circle((x_mean, y_mean), 3*d_sigma, color="green", fill=False))
    fig.tight_layout()
    plt.show()
    return current_df

