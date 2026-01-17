#!/usr/bin/env python3
"""
view_referees.py - Visualizza e cerca arbitri nel database

USO:
    python view_referees.py                    # Mostra tutti gli arbitri
    python view_referees.py "Orsato"           # Cerca arbitro specifico
    python view_referees.py --top 10           # Top 10 più severi
    python view_referees.py --bottom 10        # Top 10 più permissivi
    python view_referees.py --league "Serie A" # Solo arbitri Serie A
"""

import json
import sys
import os

DATABASE_FILE = "referee_data.json"


def load_database():
    """Carica il database arbitri"""
    if not os.path.exists(DATABASE_FILE):
        print(f"❌ Database non trovato: {DATABASE_FILE}")
        print("   Esegui prima: python build_referee_database.py")
        sys.exit(1)
    
    with open(DATABASE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def print_referee(name, stats, show_details=False):
    """Stampa info di un arbitro"""
    severity = stats["severity_factor"]
    
    # Emoji in base alla severità
    if severity > 1.15:
        emoji = "🔴"
        label = "MOLTO SEVERO"
    elif severity > 1.05:
        emoji = "🟠"
        label = "SEVERO"
    elif severity < 0.85:
        emoji = "🟢"
        label = "MOLTO PERMISSIVO"
    elif severity < 0.95:
        emoji = "🟡"
        label = "PERMISSIVO"
    else:
        emoji = "⚪"
        label = "NELLA MEDIA"
    
    print(f"\n{emoji} {name}")
    print(f"   📊 Media: {stats['avg_cards_per_match']:.2f} cartellini/partita")
    print(f"   🟨 Gialli: {stats['avg_yellow_per_match']:.2f}/partita | 🟥 Rossi: {stats['red_cards']} totali")
    print(f"   ⚖️  Severità: {severity:.3f}x ({label})")
    print(f"   🎮 Partite analizzate: {stats['matches']}")
    print(f"   🏆 Leghe: {', '.join(stats['leagues'])}")


def main():
    db = load_database()
    referees = db.get("referees", {})
    
    print("=" * 60)
    print("👨‍⚖️ DATABASE ARBITRI")
    print("=" * 60)
    print(f"📅 Ultimo aggiornamento: {db.get('last_updated', 'N/A')[:10]}")
    print(f"📊 Arbitri totali: {db.get('total_referees', 0)}")
    print(f"⚽ Media globale: {db.get('global_average_cards', 0):.2f} cartellini/partita")
    print(f"💰 Chiamate API usate: {db.get('api_calls_used', 0)}")
    
    # Medie per lega
    print("\n📈 MEDIE PER LEGA:")
    for league, stats in db.get("league_averages", {}).items():
        print(f"   {league}: {stats['avg_cards']:.2f} cart/partita ({stats['total_matches_analyzed']} partite)")
    
    # Parsing argomenti
    args = sys.argv[1:]
    
    if not args:
        # Mostra tutti ordinati per severità
        print("\n" + "=" * 60)
        print("📋 TUTTI GLI ARBITRI (ordinati per severità)")
        print("=" * 60)
        
        sorted_refs = sorted(referees.items(), key=lambda x: x[1]["severity_factor"], reverse=True)
        for name, stats in sorted_refs:
            print_referee(name, stats)
    
    elif args[0] == "--top":
        n = int(args[1]) if len(args) > 1 else 10
        print(f"\n🔴 TOP {n} ARBITRI PIÙ SEVERI:")
        print("-" * 40)
        sorted_refs = sorted(referees.items(), key=lambda x: x[1]["severity_factor"], reverse=True)
        for name, stats in sorted_refs[:n]:
            print_referee(name, stats)
    
    elif args[0] == "--bottom":
        n = int(args[1]) if len(args) > 1 else 10
        print(f"\n🟢 TOP {n} ARBITRI PIÙ PERMISSIVI:")
        print("-" * 40)
        sorted_refs = sorted(referees.items(), key=lambda x: x[1]["severity_factor"])
        for name, stats in sorted_refs[:n]:
            print_referee(name, stats)
    
    elif args[0] == "--league":
        league = args[1] if len(args) > 1 else "Serie A"
        print(f"\n⚽ ARBITRI {league.upper()}:")
        print("-" * 40)
        filtered = {k: v for k, v in referees.items() if league in v.get("leagues", [])}
        sorted_refs = sorted(filtered.items(), key=lambda x: x[1]["severity_factor"], reverse=True)
        for name, stats in sorted_refs:
            print_referee(name, stats)
        print(f"\nTotale: {len(filtered)} arbitri")
    
    else:
        # Cerca arbitro per nome
        search = " ".join(args).lower()
        print(f"\n🔍 RICERCA: '{search}'")
        print("-" * 40)
        
        found = False
        for name, stats in referees.items():
            if search in name.lower():
                print_referee(name, stats, show_details=True)
                found = True
        
        if not found:
            print(f"❌ Nessun arbitro trovato con '{search}'")
            print("\n💡 Suggerimento: prova con parte del cognome")


if __name__ == "__main__":
    main()
