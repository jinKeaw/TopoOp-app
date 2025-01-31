from tbparse import SummaryReader

log_dir = r"C:\Users\Jin\PycharmProjects\DRL_TopoOp\src\GUI\Dashboard_Data\matricslog.tfevents"
reader = SummaryReader(log_dir)
df = reader.scalars
print(df)