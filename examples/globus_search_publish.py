from garden_ai import GardenClient

garden_doc = {
    "type": "file",
    "extension": "txt",
    "name": "robots.txt"
    }

garden_id = 'foobar'

client = GardenClient()

result = client.publish_garden(garden_id, garden_doc, ['public'])
print(result)
