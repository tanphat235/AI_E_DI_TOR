"""Analysis: turning raw media into structured facts.

Speech, video and audio analysis are independent of one another and none of them
knows a renderer exists. Their only output is data, which the AI director reads and
the Edit Plan eventually references.
"""
