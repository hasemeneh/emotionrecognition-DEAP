using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EmoRecoClient
{
    public class IndividualModel
    {
		// model untuk menampung data
		string nama;
		int arousal;
		int valence;

		public string Nama { get => nama; set => nama = value; }
		public int Arousal { get => arousal; set => arousal = value; }
		public int Valence { get => valence; set => valence = value; }
		public string Summary { get {
				// ringkasan Emosi
				string summary="";
				if (arousal>0)
				{
					summary = summary + "High Arousal ";
				}
				else
				{
					summary = summary + "Low Arousal ";
				}
				if (valence > 0)
				{
					summary = summary + "High Valence ";
				}
				else
				{
					summary = summary + "Low Valence ";
				}
				return summary;
			}
		}
	}
}
