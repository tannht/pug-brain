import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Sword, Construction } from "lucide-react"

/**
 * Neurodungeon — Roguelike dungeon crawler powered by brain data.
 * Stub page while the game engine is being developed.
 */
export default function NeurodungeonPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
          <Sword className="size-8 text-primary" />
          Neurodungeon
        </h1>
        <p className="text-muted-foreground mt-1">
          Explore your memories as a roguelike dungeon crawler
        </p>
      </div>

      <Card className="group relative overflow-hidden">
        <div className="absolute inset-0 scanning-line opacity-0 group-hover:opacity-20 transition-opacity pointer-events-none" />
        <CardHeader className="relative z-10">
          <CardTitle className="flex items-center gap-2">
            <Construction className="size-5 text-yellow-500" />
            Coming Soon
          </CardTitle>
        </CardHeader>
        <CardContent className="relative z-10">
          <p className="text-muted-foreground leading-relaxed">
            Neurodungeon transforms your brain&apos;s neural graph into a
            playable dungeon. Neurons become rooms, synapses become corridors,
            and fibers become floors. Battle corrupted memories, collect items,
            and explore the depths of your knowledge graph.
          </p>
          <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
            {[
              { label: "Rooms", desc: "from neurons" },
              { label: "Corridors", desc: "from synapses" },
              { label: "Floors", desc: "from fibers" },
              { label: "Bosses", desc: "from clusters" },
            ].map((item) => (
              <div
                key={item.label}
                className="rounded-lg border bg-card p-3 text-center"
              >
                <div className="font-semibold">{item.label}</div>
                <div className="text-xs text-muted-foreground">{item.desc}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
